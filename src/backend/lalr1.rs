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
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use derivative::Derivative;
use crate::ir::grammar::{CaretExpr, Grammar, Symbol};

/// So-said "simple" types, with the tag types muted.
pub mod simple {
    /// Simple LALR entries, with no tag data.
    pub type Entry<'a, A> = super::Entry<'a, A, ()>;
    /// An LALR state set with no additional information attached to generation rules.
    pub type State<'a, A> = super::State<'a, A, ()>;
    /// An LALR kernel state set with no additional information attached to generation rules.
    pub type Kernel<'a, A> = super::Kernel<'a, A, ()>;
    /// [`FrozenKernel`] with no tag data.
    pub type FrozenKernel<'a, A> = super::FrozenKernel<'a, A, ()>;
}

/// Non-terminal symbols, represented as their index in the [`Grammar`], including the internal
/// start symbol `S` (indexed 0).
pub type NonTerminal = usize;

/// An LALR entry in a state set.
#[derive(Debug)]
#[derive(Derivative)]
#[derivative(PartialEq(bound = "Tag: PartialEq"), Eq(bound = "Tag: Eq"))]
#[derivative(PartialOrd(bound = "Tag: PartialOrd"), Ord(bound = "Tag: Ord"))]
#[derivative(Clone(bound = "Tag: Clone"), Copy(bound = "Tag: Copy"))]
pub struct Entry<'a, A, Tag> {
    /// The non-terminal symbol of this entry.
    pub symbol: NonTerminal,
    /// The RHS of the generation rule, with a caret in it for the current status.
    pub rhs: CaretExpr<'a, A>,
    /// The tag data for this entry, typically lookahead tokens or simply nothing.
    pub tag: Tag,
}

/// An LALR state set.
pub type State<'a, A, Tag> = BTreeSet<Entry<'a, A, Tag>>;
/// An LALR kernel state set (all `A -> . β` rules omitted).
/// This property is not enforced, this type is only for better readability.
pub type Kernel<'a, A, Tag> = State<'a, A, Tag>;

/// The `CLOSURE` of an item set.
pub fn closure<'a, A>(
    g: &'a Grammar<A>,
    s: impl Iterator<Item=&'a simple::Entry<'a, A>>,
) -> simple::State<A> {
    let mut res = s.copied().collect::<BTreeSet<_>>();
    let mut already_inserted = BTreeSet::new();
    let mut queue = VecDeque::new();
    queue.extend(res.iter().copied().filter_map(|e| e.rhs.next_non_terminal()));
    while let Some(x) = queue.pop_front() {
        already_inserted.insert(x);
        queue.extend(g.rules_of(x).iter()
            .map(CaretExpr::new)
            .filter(|&rhs| res.insert(Entry { symbol: x.get(), rhs: rhs, tag: () }))
            .filter_map(CaretExpr::next_non_terminal)
            .filter(|nt| !already_inserted.contains(nt)));
    }
    res
}

/// Calculate all non-empty `GOTO(I, X)` for each `X ∈ V`.
pub fn all_goto_sets<'a, A, Tag>(i: &State<'a, A, Tag>) -> BTreeMap<Symbol<A>, State<'a, A, Tag>>
    where A: Ord + Clone, Tag: Ord + Clone {
    let mut res = BTreeMap::<Symbol<A>, State<A, Tag>>::new();
    for entry in i.iter().cloned() {
        if let Some((x, rhs)) = entry.rhs.step() {
            res.entry(x.clone()).or_default().insert(Entry { rhs, ..entry });
        }
    }
    res
}

/// Kernel sets frozen for efficient access.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct FrozenKernel<'a, A, Tag>(Box<[Entry<'a, A, Tag>]>);

impl<'a, A, Tag: Ord> FrozenKernel<'a, A, Tag> {
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
    pub fn index_of(&self, i: &Kernel<A, Tag>) -> Option<usize> where Tag: Ord {
        self.kernels.binary_search_by(|k| k.compare(i)).ok()
    }
    /// Calculate the `GOTO(from, sym)` set.
    pub fn goto(&self, from: usize, sym: &Symbol<A>) -> Option<usize> where A: Ord {
        self.goto_table.binary_search_by_key(&(from, sym), |(i, x, _)| (*i, x)).ok()
    }
}

/// Calculate all kernel sets of a given [`Grammar`].
pub fn all_kernel_sets<A: Ord + Clone + Debug>(g: &Grammar<A>) -> KernelSets<A, ()> {
    let start = Rc::new(g.start_rules().iter()
        .map(CaretExpr::new)
        .map(|rhs| Entry { symbol: 0, rhs, tag: () })
        .collect::<simple::Kernel<A>>());
    let mut kernels_map = BTreeMap::new();
    kernels_map.insert(start.clone(), 0usize);
    let mut queue = VecDeque::new();
    queue.push_back((0usize, start.clone()));
    let mut goto_table = BTreeMap::new();
    while let Some((idx, i)) = queue.pop_front() {
        for (sym, states) in all_goto_sets(&i) {
            let states = Rc::new(states);
            let k = kernels_map.len();
            let k = *kernels_map.entry(states.clone()).or_insert_with(|| {
                queue.push_back((k, states));
                k
            });
            goto_table.insert((idx, sym), k);
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
