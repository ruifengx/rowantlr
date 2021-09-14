/*
 * rowantlr: ANTLR-like parser generator framework targetting rowan
 * Copyright (C) 2021  Xie Ruifeng
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

//! `LALR(1)` backend for rowantlr and related stuffs.
//!
//! ```
//! #![allow(non_snake_case)]
//! use rowantlr::r#box;
//! use rowantlr::ir::grammar::{Grammar, epsilon, Symbol::*, Expr};
//! use rowantlr::backend::ll1::{calc_deduce_to_empty, calc_first};
//! use rowantlr::backend::lalr1::closure;
//!
//! let mut g = Grammar::<&'static str>::build(|g| {
//!     let [E, T, E_, T_, F] = g.add_non_terminals();
//!     // E  —→ T E'
//!     g.mark_as_start(E);
//!     g.add_rule(E, r#box![NonTerminal(T), NonTerminal(E_)]);
//!     // E' —→ + T E' | ε
//!     g.add_rule(E_, r#box![Terminal("+"), NonTerminal(T), NonTerminal(E_)]);
//!     g.add_rule(E_, epsilon());
//!     // T  —→ F T'
//!     g.add_rule(T, r#box![NonTerminal(F), NonTerminal(T_)]);
//!     // T' —→ * F T' | ε
//!     g.add_rule(T_, r#box![Terminal("*"), NonTerminal(F), NonTerminal(T_)]);
//!     g.add_rule(T_, epsilon());
//!     // F  —→ ( E ) | id
//!     g.add_rule(F, r#box![Terminal("("), NonTerminal(E), Terminal(")")]);
//!     g.add_rule(F, r#box![Terminal("id")]);
//! });
//!
//! let deduce_to_empty = calc_deduce_to_empty(&g);
//! let first = calc_first(&g, &deduce_to_empty);
//! ```

use std::rc::Rc;
use std::fmt::Debug;
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use derivative::Derivative;
use itertools::{Itertools, GroupBy};
use crate::ir::grammar::{CaretExpr, Grammar, Symbol};

/// So-said "simple" types, with the tag types muted.
pub mod simple {
    use crate::ir::grammar::Symbol;

    /// Simple LALR entries, with no tag data.
    pub type Entry<'a, A> = super::Entry<'a, A, ()>;
    /// An LALR state set with no additional information attached to production rules.
    pub type State<'a, A> = super::State<'a, A, ()>;
    /// An LALR kernel state set with no additional information attached to production rules.
    pub type Kernel<'a, A> = super::Kernel<'a, A, ()>;
    /// [`FrozenKernel`] with no tag data.
    pub type FrozenKernel<'a, A> = super::FrozenKernel<'a, A, ()>;

    /// A trivial [`TagManager`](super::TagManager) for these simple types with no tag data.
    pub struct TagManager;

    impl<A> super::TagManager<A> for TagManager {
        type Tag = ();
        fn generate_tag(&mut self, _: &[Symbol<A>], _: &Self::Tag) -> Self::Tag {}
        fn update_tag(&mut self, _: &[Symbol<A>], _: &Self::Tag, _: &Self::Tag) {}
        fn root_tag(&mut self) -> Self::Tag {}
    }
}

/// Non-terminal symbols, represented as their index in the [`Grammar`], including the internal
/// start symbol `S` (indexed 0).
pub type NonTerminal = usize;

/// A production rule used in a state set.
#[derive(Debug)]
#[derive(Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""))]
#[derivative(PartialOrd(bound = ""), Ord(bound = ""))]
#[derivative(Clone(bound = ""), Copy(bound = ""))]
pub struct Rule<'a, A> {
    /// The non-terminal symbol of this entry.
    pub symbol: NonTerminal,
    /// The RHS of the production rule, with a caret in it for the current status.
    pub rhs: CaretExpr<'a, A>,
}

/// An LALR entry in a state set.
#[derive(Debug)]
#[derive(Derivative)]
#[derivative(PartialEq(bound = ""), Eq(bound = ""))]
#[derivative(PartialOrd(bound = ""), Ord(bound = ""))]
#[derivative(Clone(bound = "Tag: Clone"), Copy(bound = "Tag: Copy"))]
pub struct Entry<'a, A, Tag> {
    /// The production rule of this entry.
    pub rule: Rule<'a, A>,
    /// The tag data for this entry, typically lookahead tokens or simply nothing.
    #[derivative(PartialEq = "ignore")]
    #[derivative(PartialOrd = "ignore", Ord = "ignore")]
    pub tag: Tag,
}

impl<'a, A, Tag> Borrow<Rule<'a, A>> for Entry<'a, A, Tag> {
    fn borrow(&self) -> &Rule<'a, A> { &self.rule }
}

/// An LALR state set.
pub type State<'a, A, Tag> = BTreeSet<Entry<'a, A, Tag>>;
/// An LALR kernel state set (all `A -> . β` rules omitted).
/// This property is not enforced, this type is only for better readability.
pub type Kernel<'a, A, Tag> = State<'a, A, Tag>;

/// Tag managers control the behaviour of [`closure`] when an [`Entry`] is to be generated.
///
/// The same non-terminal symbol might be generated repeatedly from different production rules, and
/// when this happens, we need to somehow update the previously-generated `Tag`. However, the way
/// we store the tags makes it impossible to get mutable references to a `Tag`, so the method
/// [`TagManager::update_tag`] is only provided a shared reference.
///
/// There are two possible strategies for dealing with such situations:
///
/// - Keep all the real contents of the tags inside this `TagManager`, and use handles (possibly
///   indices) into this manager for tag types. This way when a tag is to be updated, just modify
///   the real contents in this manager, and leave the tags (handles) untouched.
/// - Wrap tag types with [`Cell`]s (when `Tag` is `Copy`) or [`RefCell`]s (when `Tag`s are heavy)
///   etc. to obtain inherent mutability. This way two `Tag`s can be properly updated.
pub trait TagManager<A> {
    /// The tag type the [`Entry`]s is about to use.
    type Tag;
    /// For the first time we see an entry shaped `A -> α . X β` (where `α, β ∈ V*` and `X ∈ VN`),
    /// this method is used to generate a new `Tag` for this entry. The tag for `A` is passed in as
    /// the `parent_tag`, and `β` is provided as `upcoming` symbols.
    fn generate_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag) -> Self::Tag;
    /// If a `Tag` is already present for an entry, and this entry is generated again possibly by
    /// some different production rule in the set, this method is used to update the old `Tag`.
    fn update_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag, previous: &Self::Tag);
    /// The entries generated from the production rule of the start symbol `S` do not have a valid
    /// parent, nor do they have upcoming symbols. `Tag`s for them are generated specially here.
    fn root_tag(&mut self) -> Self::Tag;
}

/// The `CLOSURE` of an item set.
pub fn closure<'a, A, M: TagManager<A>>(
    g: &'a Grammar<A>,
    s: impl Iterator<Item=Entry<'a, A, M::Tag>>,
    mgr: &mut M,
) -> State<'a, A, M::Tag> where M::Tag: Clone + 'a {
    let mut res = s.collect::<BTreeSet<_>>();
    let mut queue = VecDeque::new();
    queue.extend(res.iter().map(|entry| entry.rule)
        .filter(|rule| rule.rhs.step_non_terminal().is_some()));
    while let Some(parent_rule) = queue.pop_front() {
        let (x, upcoming) = parent_rule.rhs.step_non_terminal().unwrap();
        let upcoming = upcoming.rest_part();
        for rhs in g.rules_of(x).iter().map(CaretExpr::new) {
            let rule = Rule { symbol: x.get(), rhs };
            let parent_tag = &res.get(&parent_rule).unwrap().tag;
            if let Some(Entry { tag: previous, .. }) = res.get(&rule) {
                mgr.update_tag(upcoming, parent_tag, previous);
            } else {
                let tag = mgr.generate_tag(upcoming, parent_tag);
                res.insert(Entry { rule, tag });
                if rule.rhs.step_non_terminal().is_some() {
                    queue.push_back(rule);
                }
            }
        }
    }
    res
}

type EntryIter<'a, A, Tag> = std::vec::IntoIter<(&'a Symbol<A>, Entry<'a, A, Tag>)>;
type SndFn<'a, A, Tag> = fn(&(&'a Symbol<A>, Entry<'a, A, Tag>)) -> &'a Symbol<A>;

/// Iterables for traversing `GOTO` sets, always call [`RawGotoSets::get`] on this type.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct RawGotoSets<'a, A, Tag>(GroupBy<&'a Symbol<A>, EntryIter<'a, A, Tag>, SndFn<'a, A, Tag>>);

impl<'a, A: 'a + PartialEq, Tag> RawGotoSets<'a, A, Tag> {
    /// Get the iterable for the `GOTO` sets.
    pub fn get<'s>(&'s self) -> impl Iterator<Item=(&'a Symbol<A>, impl Iterator<Item=Entry<'a, A, Tag>> + 's)> + 's where 'a: 's {
        self.0.into_iter().map(|(x, group)| (x, group.map(|(_, entry)| entry)))
    }
}

/// Calculate all non-empty `GOTO(I, X)` for each `X ∈ V`.
pub fn all_goto_sets<'s, 'a, A: 'a + PartialEq + Ord + Clone, Tag: 'a + Clone>(
    i: &Kernel<'a, A, Tag>) -> RawGotoSets<'a, A, Tag> {
    let mut res = i.iter().cloned().filter_map(|mut entry| {
        let (x, rhs) = entry.rule.rhs.step()?;
        entry.rule.rhs = rhs;
        Some((x, entry))
    }).collect_vec();
    res.sort_unstable();
    RawGotoSets(res.into_iter().group_by(|p| p.0))
}

/// Kernel sets frozen for efficient access.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct FrozenKernel<'a, A, Tag>(Box<[Entry<'a, A, Tag>]>);

impl<'a, A, Tag> FrozenKernel<'a, A, Tag> {
    /// Freeze a [`Kernel`] for future use.
    pub fn freeze(i: Kernel<'a, A, Tag>) -> Self {
        FrozenKernel(i.into_iter().collect())
    }

    /// Compare with a [`Kernel`].
    pub fn compare(&self, i: &Kernel<'a, A, Tag>) -> Ordering {
        self.0.iter().cmp(i.iter())
    }

    /// Get all the rules in this kernel set.
    pub fn rules(&self) -> impl Iterator<Item=&Entry<'a, A, Tag>> {
        self.0.iter()
    }
}

/// All kernel sets for a [`Grammar`], together with the full `GOTO` table.
pub struct KernelSets<'a, A, Tag> {
    kernels: Box<[FrozenKernel<'a, A, Tag>]>,
    goto_table: Box<[(usize, Symbol<A>, usize)]>,
}

impl<'a, A, Tag> KernelSets<'a, A, Tag> {
    /// Get the (ascending) list of frozen kernel sets.
    pub fn kernels(&self) -> &[FrozenKernel<'a, A, Tag>] { &self.kernels }
    /// Get index of a (non-frozen) kernel sets.
    pub fn index_of(&self, i: &Kernel<A, Tag>) -> Option<usize> {
        self.kernels.binary_search_by(|k| k.compare(i)).ok()
    }
    /// Calculate the `GOTO(from, sym)` set.
    pub fn goto(&self, from: usize, sym: &Symbol<A>) -> Option<usize> where A: Ord {
        self.goto_table.binary_search_by_key(&(from, sym), |(i, x, _)| (*i, x)).ok()
    }
}

/// Calculate all kernel sets of a given [`Grammar`].
pub fn all_kernel_sets<'a, A, M>(g: &'a Grammar<A>, mgr: &mut M) -> KernelSets<'a, A, M::Tag>
    where A: Ord + Clone + Debug, M: TagManager<A>, M::Tag: 'a + Debug + Clone {
    let root_tag = mgr.root_tag();
    let start = Rc::new(g.start_rules().iter()
        .map(CaretExpr::new)
        .map(|rhs| Entry { rule: Rule { symbol: 0, rhs }, tag: root_tag.clone() })
        .collect::<Kernel<A, M::Tag>>());
    let mut kernels_map = BTreeMap::new();
    kernels_map.insert(start.clone(), 0usize);
    let mut queue = VecDeque::new();
    queue.push_back((0usize, start.clone()));
    let mut goto_table = BTreeMap::new();
    while let Some((idx, i)) = queue.pop_front() {
        for (sym, states) in all_goto_sets(&i).get() {
            let states = Rc::new(closure(g, states, mgr));
            let k = kernels_map.len();
            let k = *kernels_map.entry(states.clone()).or_insert_with(|| {
                queue.push_back((k, states));
                k
            });
            goto_table.insert((idx, sym.clone()), k);
        }
    }
    let mut kernels = Vec::with_capacity(kernels_map.len());
    let mut idx_map = vec![0usize; kernels_map.len()];
    for (new_idx, (kernel, old_idx)) in kernels_map.into_iter().enumerate() {
        idx_map[old_idx] = new_idx;
        kernels.push(FrozenKernel::freeze(Rc::try_unwrap(kernel).unwrap()));
    }
    let goto_table = goto_table.into_iter()
        .map(|((from, sym), to)| (idx_map[from], sym, idx_map[to]))
        .collect::<Box<[_]>>();
    KernelSets { kernels: kernels.into_boxed_slice(), goto_table }
}
