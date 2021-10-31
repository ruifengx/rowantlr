/*
 * rowantlr: ANTLR-like parser generator framework targeting rowan
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
//! Prerequisites (as illustrated in module [`ll1`](super::ll1)):
//! - use the [`Grammar`] API to create new grammars.
//! - use [`calc_deduce_to_empty`](super::ll1::calc_deduce_to_empty) to calculate `DEDUCE_TO_EMPTY`.
//! - use [`calc_first`](super::ll1::calc_first) to calculate `FIRST`.
//!
//! Call [`build`] to generate `ACTION` and `GOTO` tables, and use [`Table::simulate_parse`] to
//! test the grammar on some token streams.
//!
//! ```
//! #![allow(non_snake_case)]
//! use std::convert::TryFrom;
//! use rowantlr::r#box;
//! use rowantlr::ir::syntax::{Grammar, epsilon, Symbol::*, Expr};
//! use rowantlr::backend::{ll1, ll1::Lookahead, lalr1, lalr1::Action::*};
//! # use rowantlr::utils::DisplayDot2TeX;
//!
//! let mut g = Grammar::<&'static str>::build(|g| {
//!     let [S, L, R] = g.add_non_terminals();
//!     // S —→ L = R | R
//!     g.mark_as_start(S);
//!     g.add_rule(S, r#box![NonTerminal(L), Terminal("="), NonTerminal(R)]);
//!     g.add_rule(S, r#box![NonTerminal(R)]);
//!     // L —→ * R | id
//!     g.add_rule(L, r#box![Terminal("*"), NonTerminal(R)]);
//!     g.add_rule(L, r#box![Terminal("id")]);
//!     // R —→ L
//!     g.add_rule(R, r#box![NonTerminal(L)]);
//! });
//!
//! let deduce_to_empty = ll1::calc_deduce_to_empty(&g);
//! let first = ll1::calc_first(&g, &deduce_to_empty);
//! let kernels = lalr1::build(&g, &first, &deduce_to_empty);
//! # println!("{}", kernels.display_dot2tex(&["S'", "S", "L", "R"]));
//! let table = lalr1::Table::try_from(&kernels).ok().unwrap();
//! // correctly parse "* * id = id"
//! let parse = table.simulate_parse(["*", "*", "id", "=", "id"]).unwrap();
//! assert_eq!(
//!     format!("{}", parse.pretty(&["S'", "S", "L", "R"])),
//!     indoc::indoc! {r#"
//!         NT#[ S ]
//!           NT#[ L ]
//!             "*"
//!             NT#[ R ]
//!               NT#[ L ]
//!                 "*"
//!                 NT#[ R ]
//!                   NT#[ L ]
//!                     "id"
//!           "="
//!           NT#[ R ]
//!             NT#[ L ]
//!               "id"
//!     "#});
//! // cannot parse "* * id id" (expecting "=" or EOF)
//! let err = table.simulate_parse(["*", "*", "id", "id"]).unwrap_err();
//! assert_eq!(err.unexpected, Lookahead::new("id"));
//! assert_eq!(err.expecting, r#box![Lookahead::END_OF_INPUT, Lookahead::new("=")]);
//! // cannot parse "* * id =" (expecting "id" or "*")
//! let err = table.simulate_parse(["*", "*", "id", "="]).unwrap_err();
//! assert_eq!(err.unexpected, Lookahead::END_OF_INPUT);
//! assert_eq!(err.expecting, r#box![Lookahead::new("*"), Lookahead::new("id")]);
//! ```
//!
//! To visualise the generated LALR automaton, run the following:
//!
//! ```no_run
//! # use rowantlr::backend::lalr1::{KernelSets, Lookaheads};
//! # use rowantlr::utils::DisplayDot2TeX;
//! # let kernels: KernelSets<'static, &'static str, Lookaheads<&'static str>> = unimplemented!();
//! println!("{}", kernels.display_dot2tex(&["S'", "S", "L", "R"]));
//! ```
//!
//! and typeset the generated contents using `doc2tex` and LaTeX.

use std::rc::Rc;
use std::fmt::{Debug, Display, Formatter};
use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use derivative::Derivative;
use itertools::{Itertools, GroupBy};
use crate::ir::syntax::{CaretExpr, Grammar, Symbol};
use crate::backend::ll1::Lookahead;
use crate::utils::{Dict, DisplayDot2TeX, simple::DisplayDot2TeX as DisplayDot2TeX_};
use std::convert::TryFrom;

/// So-said "simple" types, with the tag types muted.
pub mod simple {
    use crate::ir::syntax::Symbol;

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
        fn merge_tags<'a>(&mut self, _: impl Iterator<Item=(&'a Self::Tag, &'a Self::Tag)>) {}
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

impl<'a, A: Display> Display for Rule<'a, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(NT#{}) -> {}", self.symbol, self.rhs)
    }
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

impl<'a, A: Display, Tag: Display> Display for Entry<'a, A, Tag> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} //{}", self.rule, self.tag)
    }
}

impl<'a, A: DisplayDot2TeX, Tag: DisplayDot2TeX> DisplayDot2TeX for Entry<'a, A, Tag> {
    fn fmt_dot2tex(&self, _: &'_ (), f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, r#"        \(\NT{{{}}} \to"#, self.rule.symbol)?;
        self.rule.rhs.fmt_dot2tex_(f)?;
        writeln!(f, r#"\), & \({}\) \cr"#, self.tag.display_dot2tex_())
    }
}

impl<'a, A, Tag, T> DisplayDot2TeX<[T]> for Entry<'a, A, Tag>
    where A: DisplayDot2TeX<[T]>, Tag: DisplayDot2TeX<[T]>, T: AsRef<str> {
    fn fmt_dot2tex(&self, dict: &'_ [T], f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, r#"        \(\NT{{{}}} \to"#, dict[self.rule.symbol].as_ref().display_dot2tex_())?;
        self.rule.rhs.fmt_dot2tex(dict, f)?;
        writeln!(f, r#"\), & \({}\) \cr"#, self.tag.display_dot2tex(dict))
    }
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
/// - Wrap tag types with [`Cell`](std::cell::Cell)s (when `Tag` is `Copy`) or
///   [`RefCell`](std::cell::RefCell)s (when `Tag`s are heavy) etc. to obtain inherent mutability.
///   This way two `Tag`s can be properly updated.
pub trait TagManager<A> {
    /// The tag type the [`Entry`]s is about to use.
    type Tag: 'static;
    /// For the first time we see an entry shaped `A -> α . X β` (where `α, β ∈ V*` and `X ∈ VN`),
    /// this method is used to generate a new `Tag` for this entry. The tag for `A` is passed in as
    /// the `parent_tag`, and `β` is provided as `upcoming` symbols.
    fn generate_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag) -> Self::Tag;
    /// If a `Tag` is already present for an entry, and this entry is generated again possibly by
    /// some different production rule in the set, this method is used to update the old `Tag`.
    fn update_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag, previous: &Self::Tag);
    /// After generating the tags for a state set, we might discover that the set is duplicated.
    /// In this case this method is used to merge the tags from the two sets, and feel free to
    /// remove those generated tags.
    ///
    /// The tags are presented as a series of pairs.
    fn merge_tags<'a>(&mut self, tags: impl Iterator<Item=(&'a Self::Tag, &'a Self::Tag)>);
    /// The entries generated from the production rule of the start symbol `S` do not have a valid
    /// parent, nor do they have upcoming symbols. `Tag`s for them are generated specially here.
    fn root_tag(&mut self) -> Self::Tag;
}

/// The `CLOSURE` of an item set.
pub fn closure<'a, A, M: TagManager<A>>(
    g: &'a Grammar<A>, s: Kernel<'a, A, M::Tag>, mgr: &mut M,
) -> State<'a, A, M::Tag> where M::Tag: Clone + 'a {
    let mut res = s;
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

type EntryIter<'a, A, Tag> = itertools::Dedup<std::vec::IntoIter<(&'a Symbol<A>, Entry<'a, A, Tag>)>>;
type SndFn<'a, A, Tag> = fn(&(&'a Symbol<A>, Entry<'a, A, Tag>)) -> &'a Symbol<A>;

/// Iterables for traversing `GOTO` sets, always call [`RawGotoSets::get`] on this type.
#[must_use = "iterator adaptors are lazy and do nothing unless consumed"]
pub struct RawGotoSets<'a, A: PartialEq, Tag>(GroupBy<
    &'a Symbol<A>,
    EntryIter<'a, A, Tag>,
    SndFn<'a, A, Tag>
>);

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
    RawGotoSets(res.into_iter().dedup().group_by(|p| p.0))
}

/// Kernel sets frozen for efficient access.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct FrozenKernel<'a, A, Tag>(Box<[Entry<'a, A, Tag>]>);

impl<'a, A: Display, Tag: Display> Display for FrozenKernel<'a, A, Tag> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for entry in self.0.iter() {
            writeln!(f, "{}", entry)?;
        }
        Ok(())
    }
}

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
#[derive(Debug)]
pub struct KernelSets<'a, A, Tag> {
    kernels: Box<[FrozenKernel<'a, A, Tag>]>,
    goto_table: Dict<(usize, Symbol<A>, usize)>,
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
        self.goto_table.get((&from, sym)).copied()
    }
}

impl<'a, A: Display, Tag: Display> Display for KernelSets<'a, A, Tag> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (k, ker) in self.kernels.iter().enumerate() {
            writeln!(f, "State #{}:", k)?;
            writeln!(f, "{}", ker)?;
        }
        for (from, sym, to) in self.goto_table.iter() {
            writeln!(f, "{:16} = {}", format!("GOTO({}, {})", from, sym), to)?;
        }
        Ok(())
    }
}

const DOT2TEX_TEMPLATE_START: &str = indoc::indoc! {r#"
    \documentclass{standalone}
    \usepackage{multicol}
    \usepackage{tikz}
    \usetikzlibrary{automata, positioning, arrows}
    \usepackage{dot2texi}

    \colorlet{token}{lightgray!50!white}
    \def\token#1{\colorbox{token}{\tofullheight{#1}}}
    \def\NT#1{\langle\mbox{#1}\rangle}
    \def\caret{{{{}\cdot{}}}}
    \def\EOF{\#}

    \newlength{\fullheight}
    \newlength{\fulldepth}
    \settoheight{\fullheight}{(}
    \settodepth{\fulldepth}{(}
    \def\tofullheight#1{\raisebox{0pt}[\fullheight][\fulldepth]{#1}}

    \begin{document}
    \def\arraystretch{1.5}
    \begin{dot2tex}[styleonly,options={-t raw}]
    digraph G {
        node[style="inner sep=0pt"];
"#};
const DOT2TEX_TEMPLATE_END: &str = indoc::indoc! {r#"
    }
    \end{dot2tex}
    \end{document}
"#};

impl<'a, A, Tag, T> DisplayDot2TeX<[T]> for KernelSets<'a, A, Tag>
    where A: DisplayDot2TeX<[T]>, Tag: DisplayDot2TeX<[T]>, T: AsRef<str> {
    fn fmt_dot2tex(&self, dict: &[T], f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", DOT2TEX_TEMPLATE_START)?;
        for (k, ker) in self.kernels.iter().enumerate() {
            writeln!(f, r#"    q{}[label="\begin{{tabular}}{{|ll|}}\hline"#, k)?;
            writeln!(f, r#"        \multicolumn{{2}}{{|l|}}{{State {}:}}\cr\hline"#, k)?;
            for entry in ker.0.iter() {
                entry.fmt_dot2tex(dict, f)?;
            }
            writeln!(f, r#"    \hline\end{{tabular}}"];"#)?;
        }
        for (from, sym, to) in self.goto_table.iter() {
            writeln!(f, r#"    q{} -> q{} [label="{}"]"#, from, to, sym.display_dot2tex(dict))?;
        }
        writeln!(f, "{}", DOT2TEX_TEMPLATE_END)
    }
}

/// Calculate all kernel sets of a given [`Grammar`].
pub fn all_kernel_sets<'a, A, M>(g: &'a Grammar<A>, mgr: &mut M) -> KernelSets<'a, A, M::Tag>
    where A: Ord + Clone, M: TagManager<A>, M::Tag: Clone + 'a {
    let root_tag = mgr.root_tag();
    let start = Rc::new(closure(g, g.start_rules().iter()
        .map(CaretExpr::new)
        .map(|rhs| Entry { rule: Rule { symbol: 0, rhs }, tag: root_tag.clone() })
        .collect::<Kernel<A, M::Tag>>(), mgr));
    let mut kernels_map = BTreeMap::new();
    kernels_map.insert(start.clone(), 0usize);
    let mut queue = VecDeque::new();
    queue.push_back((0usize, start));
    let mut goto_table = BTreeMap::new();
    while let Some((idx, i)) = queue.pop_front() {
        for (sym, states) in all_goto_sets(&i).get() {
            let states = Rc::new(closure(g, states.map(|entry| Entry {
                tag: mgr.generate_tag(&[], &entry.tag),
                rule: entry.rule,
            }).collect::<Kernel<A, M::Tag>>(), mgr));
            use std::collections::btree_map::Entry::*;
            let k = kernels_map.len();
            let k = match kernels_map.entry(states.clone()) {
                Vacant(to_insert) => {
                    queue.push_back((k, states.clone()));
                    to_insert.insert(k);
                    k
                }
                Occupied(existing) => {
                    mgr.merge_tags(existing.key().iter().map(|entry| &entry.tag)
                        .zip_eq(states.iter().map(|entry| &entry.tag)));
                    *existing.get()
                }
            };
            goto_table.insert((idx, sym.clone()), k);
        }
    }
    let mut kernels = Vec::with_capacity(kernels_map.len());
    let mut idx_map = vec![0usize; kernels_map.len()];
    for (new_idx, (kernel, old_idx)) in kernels_map.into_iter().enumerate() {
        idx_map[old_idx] = new_idx;
        kernels.push(FrozenKernel::freeze(Rc::try_unwrap(kernel).ok().unwrap()));
    }
    let goto_table = goto_table.into_iter()
        .map(|((from, sym), to)| (idx_map[from], sym, idx_map[to]))
        .collect();
    KernelSets { kernels: kernels.into_boxed_slice(), goto_table }
}

/// Tags in LALR state sets can be resolved to eliminate their inter-dependency.
pub trait TagResolver<A> {
    /// After resolving a tag, we get a `Resolved` to be stored back into that [`Entry`].
    type Resolved: Clone;
    /// Total number of tags.
    fn tag_count(&self) -> usize;
    /// Get the tags which should be resolved prior to the current one.
    fn get_predecessors(&self, tag: usize) -> Vec<usize>;
    /// After the dependencies have been worked out, resolve a group of `Tag`s where their
    /// dependency graph is strongly connected (contains a loop).
    fn resolve_group(&mut self, tags: impl Iterator<Item=usize>,
                     predecessors: impl Iterator<Item=Self::Resolved>) -> Self::Resolved;
}

#[derive(Derivative)]
#[derivative(Default(bound = ""))]
#[derivative(Debug(bound = "R::Resolved: Debug"))]
#[derivative(PartialEq(bound = "R::Resolved: PartialEq"), Eq(bound = "R::Resolved: Eq"))]
#[derivative(PartialOrd(bound = "R::Resolved: PartialOrd"), Ord(bound = "R::Resolved: Ord"))]
#[derivative(Clone(bound = "R::Resolved: Clone"), Copy(bound = "R::Resolved: Copy"))]
struct VertexInfo<A, R: TagResolver<A>> {
    visit_time: usize,
    earliest_reachable: usize,
    visited: bool,
    in_current_path: bool,
    resolved: Option<R::Resolved>,
}

struct Tarjan<A, R: TagResolver<A>> {
    vertices: Vec<VertexInfo<A, R>>,
    current_path: Vec<usize>,
    current_time: usize,
}

impl<A, R: TagResolver<A>> Tarjan<A, R> {
    fn new(n: usize) -> Self {
        Tarjan {
            vertices: vec![VertexInfo::default(); n],
            current_path: Vec::new(),
            current_time: 0,
        }
    }

    fn run(mut self, resolver: &mut R, tags: impl Iterator<Item=usize>) -> Box<[Option<R::Resolved>]> {
        for tag in tags {
            if !self.vertices[tag].visited {
                self.visit(resolver, tag);
            }
        }
        self.vertices.into_iter().map(|info| info.resolved).collect()
    }

    fn visit(&mut self, resolver: &mut R, this: usize) {
        let k = self.current_path.len();
        self.vertices[this].visited = true;
        self.vertices[this].in_current_path = true;
        self.vertices[this].visit_time = self.current_time;
        self.vertices[this].earliest_reachable = self.current_time;
        self.current_time += 1;
        self.current_path.push(this);

        for that in resolver.get_predecessors(this) {
            if !self.vertices[that].visited {
                self.visit(resolver, that);
                self.vertices[this].earliest_reachable = std::cmp::min(
                    self.vertices[this].earliest_reachable,
                    self.vertices[that].earliest_reachable);
            } else if self.vertices[that].in_current_path {
                self.vertices[this].earliest_reachable = std::cmp::min(
                    self.vertices[this].earliest_reachable,
                    self.vertices[that].visit_time);
            }
        }

        if self.vertices[this].visit_time == self.vertices[this].earliest_reachable {
            let group = self.current_path.split_off(k);
            let ps = resolver.get_predecessors(this)
                .into_iter().filter_map(|p| self.vertices[p].resolved.as_ref()).cloned();
            let r = resolver.resolve_group(group.iter().copied(), ps);
            for t in group {
                self.vertices[t].resolved = Some(r.clone());
            }
        }

        self.vertices[this].in_current_path = false;
    }
}

/// Resolve the tags with a [`TagResolver`] in case their is inter-dependencies among them.
///
/// This process is used calculate lookahead tokens for `KernelSets`.
pub fn resolve_tags<'a, A, R>(sets: KernelSets<'a, A, usize>, resolver: &mut R) -> KernelSets<'a, A, R::Resolved>
    where R: TagResolver<A>, R::Resolved: Clone + Debug {
    let KernelSets { kernels, goto_table } = sets;
    let tarjan = Tarjan::new(resolver.tag_count());
    let resolved = tarjan.run(resolver, kernels.iter()
        .flat_map(|k| k.0.iter())
        .map(|entry| entry.tag));
    KernelSets {
        kernels: kernels.into_vec().into_iter()
            .map(|ker| FrozenKernel(
                ker.0.into_vec().into_iter().map(|entry| Entry {
                    rule: entry.rule,
                    tag: resolved[entry.tag].as_ref().unwrap().clone(),
                }).collect())).collect(),
        goto_table,
    }
}

#[derive(Debug)]
#[derive(Derivative)]
#[derivative(Default(bound = "A: Ord"))]
struct LookaheadsWithDep<A> {
    spontaneous: BTreeSet<Lookahead<A>>,
    depends_on: BTreeSet<usize>,
}

struct LookaheadManager<'a, A> {
    deduce_to_empty: &'a [bool],
    first: &'a [Box<[A]>],
    tags: Vec<LookaheadsWithDep<A>>,
}

impl<'a, A> LookaheadManager<'a, A> {
    fn new(first: &'a [Box<[A]>], deduce_to_empty: &'a [bool]) -> Self {
        LookaheadManager { first, deduce_to_empty, tags: Vec::new() }
    }

    fn into_resolver(self) -> LookaheadResolver<A> {
        let n = self.tags.len();
        LookaheadResolver {
            tags: self.tags,
            index_map: r#box![0; n],
            resolved: Vec::new(),
        }
    }
}

impl<'a, A: Ord + Clone> TagManager<A> for LookaheadManager<'a, A> {
    type Tag = usize;

    fn generate_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag) -> Self::Tag {
        let this = self.tags.len();
        self.tags.push(LookaheadsWithDep::default());
        self.update_tag(upcoming, parent_tag, &this);
        this
    }

    fn update_tag(&mut self, upcoming: &[Symbol<A>], parent_tag: &Self::Tag, previous: &Self::Tag) {
        if super::ll1::append_first_of::<_, _, _, _, [A]>(
            upcoming, self.first, self.deduce_to_empty,
            &mut self.tags[*previous].spontaneous, &mut false) {
            self.tags[*previous].depends_on.insert(*parent_tag);
        }
    }

    fn merge_tags<'t>(&mut self, tags: impl Iterator<Item=(&'t Self::Tag, &'t Self::Tag)>) {
        let mut tags = tags.map(|(&t, &s)| (t, s)).collect_vec();
        tags.sort_unstable_by_key(|&(_, s)| s);
        let map_idx = |x: usize| tags
            .binary_search_by_key(&x, |&(_, s)| s)
            .map_or(x, |pos| tags[pos].0);

        let mut min = usize::MAX;
        let mut max = usize::MIN;
        let mut count = 0;
        for &(target, source) in &tags {
            min = std::cmp::min(min, source);
            max = std::cmp::max(max, source);
            count += 1;

            assert_ne!(target, source);
            let spontaneous = self.tags[source].spontaneous.iter().cloned().collect_vec();
            self.tags[target].spontaneous.extend(spontaneous.into_iter());
            let depends_on = self.tags[source].depends_on.iter().copied().collect_vec();
            self.tags[target].depends_on.extend(depends_on.into_iter().map(map_idx));
        }

        if max < min { return; }
        // INVARIANT: we are removing the last `count` values.
        assert_eq!(count, max - min + 1);
        assert_eq!(max, self.tags.len() - 1);
        self.tags.truncate(min);
    }

    fn root_tag(&mut self) -> Self::Tag {
        let this = self.tags.len();
        let mut tag = LookaheadsWithDep::default();
        tag.spontaneous.insert(Lookahead::END_OF_INPUT);
        self.tags.push(tag);
        this
    }
}

struct LookaheadResolver<A> {
    tags: Vec<LookaheadsWithDep<A>>,
    index_map: Box<[usize]>,
    resolved: Vec<Rc<[Lookahead<A>]>>,
}

/// A set of [`Lookahead`] tokens.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Lookaheads<A>(pub Rc<[Lookahead<A>]>);

impl<A: Display> Display for Lookaheads<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().format(", "))
    }
}

impl<A: DisplayDot2TeX<Env>, Env: ?Sized> DisplayDot2TeX<Env> for Lookaheads<A> {
    fn fmt_dot2tex(&self, env: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().map(|x| x.display_dot2tex(env)).format(r"\,"))
    }
}

impl<A: Ord + Clone> TagResolver<A> for LookaheadResolver<A> {
    type Resolved = Lookaheads<A>;

    fn tag_count(&self) -> usize { self.tags.len() }

    fn get_predecessors(&self, tag: usize) -> Vec<usize> {
        self.tags[tag].depends_on.iter().copied().collect_vec()
    }

    fn resolve_group<'a>(&mut self, tags: impl Iterator<Item=usize>,
                         predecessors: impl Iterator<Item=Self::Resolved>) -> Self::Resolved {
        let mut resolved = BTreeSet::new();
        for p in predecessors {
            resolved.extend(p.0.iter().cloned());
        }
        let current = self.resolved.len();
        for tag in tags {
            resolved.extend(self.tags[tag].spontaneous.iter().cloned());
            for &p in &self.tags[tag].depends_on {
                self.index_map[tag] = current;
                let p = self.index_map[p];
                if p != current {
                    resolved.extend(self.resolved[p].iter().cloned());
                }
            }
        }
        let resolved: Rc<[_]> = resolved.into_iter().collect_vec().into();
        self.resolved.push(resolved.clone());
        Lookaheads(resolved)
    }
}

/// Build LALR kernel sets table, and the `GOTO` table.
pub fn build<'a, A>(grammar: &'a Grammar<A>, first: &[Box<[A]>], deduce_to_empty: &[bool])
                    -> KernelSets<'a, A, Lookaheads<A>>
    where A: Debug + Ord + Clone + 'a {
    let mut tag_manager = LookaheadManager::new(first, deduce_to_empty);
    let sets = all_kernel_sets(grammar, &mut tag_manager);
    let mut resolver = tag_manager.into_resolver();
    resolve_tags(sets, &mut resolver)
}

/// An entry of the `ACTION` table.
///
/// A missing entry denotes an error.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Action {
    /// Accepts the input.
    Accept,
    /// Shift in the next token, and transition to another state.
    Shift(u32),
    /// Reduce according to a production rule.
    Reduce {
        /// Number of fragment parse trees to be popped from the stack.
        ///
        /// It is highly unlikely that any non-terminal symbol should be reduced from more than
        /// 6,5536 fragment parse trees (which in turn would mean there is a production rule with
        /// more than 6,5536 symbols as RHS).
        count: u16,
        /// The non-terminal symbol to be reduced to.
        ///
        /// It is highly unlikely that any non-terminal symbol should be represented by a integer
        /// greater than 6,5536 (which in turn would mean there are more than 6,5536 of non-terminal
        /// symbols in total, and thus there is no chance this grammar would ever be useful for any
        /// practical parsing).
        symbol: u16,
    },
}

/// The LALR parsing table.
#[derive(Debug, Eq, PartialEq)]
pub struct Table<A> {
    /// `ACTION` table: entries `(state, input, action)`.
    pub action: Dict<(u32, Lookahead<A>, Action)>,
    /// `GOTO` table: entries `(state, non-terminal, state')`.
    pub goto: Dict<(u32, u16, u32)>,
}

/// Conflicts in LALR kernel sets.
pub struct LalrConflict<'a, A> {
    /// Is there any shift-reduce conflict?
    pub shift_reduce_conflict: bool,
    /// Is there any reduce-reduce conflict?
    pub reduce_reduce_conflict: bool,
    /// The problematic kernel set.
    pub kernel_set: &'a FrozenKernel<'a, A, Lookaheads<A>>,
}

impl<'a, A, Tag> KernelSets<'a, A, Tag> {
    /// Get the corresponding [`Action`] for a production rule.
    pub fn action_of(&self, rule: &Rule<'a, A>) -> Option<Action> {
        if rule.rhs.rest_part().is_empty() {
            Some(if rule.symbol == 0 { Action::Accept } else {
                Action::Reduce {
                    symbol: rule.symbol as u16,
                    count: rule.rhs.consumed_part().len() as u16,
                }
            })
        } else {
            None
        }
    }
}

struct ActionInserter<'s, 'a, A> {
    target_actions: &'s mut BTreeMap<Lookahead<A>, Action>,
    current_kernel: &'a FrozenKernel<'a, A, Lookaheads<A>>,
    error: Option<LalrConflict<'a, A>>,
}

impl<'s, 'a, A: Ord> ActionInserter<'s, 'a, A> {
    fn new(target_actions: &'s mut BTreeMap<Lookahead<A>, Action>,
           current_kernel: &'a FrozenKernel<'a, A, Lookaheads<A>>) -> Self {
        ActionInserter { target_actions, current_kernel, error: None }
    }

    fn insert(&mut self, lookahead: Lookahead<A>, act: Action) {
        if let Some(old_act) = self.target_actions.insert(lookahead, act) {
            if matches!(old_act, Action::Shift(_)) && matches!(act, Action::Shift(_)) { return; }
            let err = if let Some(err) = &mut self.error { err } else {
                self.error = Some(LalrConflict {
                    kernel_set: self.current_kernel,
                    reduce_reduce_conflict: false,
                    shift_reduce_conflict: false,
                });
                self.error.as_mut().unwrap()
            };
            match (old_act, act) {
                (Action::Shift(_), Action::Reduce { .. }) |
                (Action::Reduce { .. }, Action::Shift(_)) => err.shift_reduce_conflict = true,
                (Action::Reduce { .. }, Action::Reduce { .. }) => err.reduce_reduce_conflict = true,
                _ => unreachable!(),
            }
        }
    }

    fn into_error(self) -> Option<LalrConflict<'a, A>> { self.error }
}

impl<'a, A: Ord + Clone> TryFrom<&'a KernelSets<'a, A, Lookaheads<A>>> for Table<A> {
    type Error = Vec<LalrConflict<'a, A>>;

    fn try_from(sets: &'a KernelSets<'a, A, Lookaheads<A>>) -> Result<Table<A>, Self::Error> {
        let mut action = Vec::new();
        let mut goto = Vec::new();
        let mut errs = Vec::new();
        for (k, ker) in sets.kernels.iter().enumerate() {
            let mut actions = BTreeMap::new();
            let mut inserter = ActionInserter::new(&mut actions, ker);
            for entry in ker.0.iter() {
                if let Some(act) = sets.action_of(&entry.rule) {
                    for x in entry.tag.0.iter() {
                        inserter.insert(x.clone(), act);
                    }
                } else {
                    for (sym, &t) in sets.goto_table.equal_range((&k, )) {
                        let t = t as u32;
                        match sym {
                            Symbol::Terminal(a) => inserter.insert(Lookahead::new(a.clone()), Action::Shift(t)),
                            Symbol::NonTerminal(nt) => goto.push((k as u32, nt.get() as u16, t)),
                        }
                    }
                }
            }
            if let Some(err) = inserter.into_error() { errs.push(err) }
            action.extend(actions.into_iter().map(|(a, act)| (k as u32, a, act)));
        }
        if !errs.is_empty() { Err(errs) } else {
            Ok(Table {
                action: action.into(),
                goto: goto.into(),
            })
        }
    }
}

/// Simulation of `LALR(1)` parsing.
pub mod simulation {
    use std::fmt::{Display, Formatter};
    use itertools::Itertools;
    use crate::backend::ll1::Lookahead;
    use crate::backend::lalr1::Action;

    /// Parse tree: result of simulation of `LALR(1)`.
    #[derive(Debug, Clone)]
    pub enum Parse<A> {
        /// Terminals: leaves of a [`Parse`] tree.
        Terminal(A),
        /// Non-terminals: internal nodes of a [`Parse`] tree.
        NonTerminal {
            /// The non-terminal symbol of this node.
            symbol: u16,
            /// Children nodes, reduced according to the production rules.
            children: Box<[Parse<A>]>,
        },
    }

    impl<A: Display> Parse<A> {
        /// Pretty print this parse tree, provided names for the non-terminal symbols.
        pub fn pretty<'a, T>(&'a self, dict: &'a [T]) -> Pretty<'a, A, T> {
            Pretty(self, dict)
        }
    }

    /// Pretty print for parse trees.
    pub struct Pretty<'a, A, T> (&'a Parse<A>, &'a [T]);

    impl<'a, A: Display, T: AsRef<str>> Display for Pretty<'a, A, T> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            self.fmt_indent(f, 0)
        }
    }

    impl<'a, A: Display, T: AsRef<str>> Pretty<'a, A, T> {
        fn fmt_indent(&self, f: &mut Formatter<'_>, indent: usize) -> std::fmt::Result {
            write!(f, "{:width$}", "", width = indent * 2)?;
            match self.0 {
                Parse::Terminal(a) => writeln!(f, "\"{}\"", a),
                Parse::NonTerminal { symbol, children } => {
                    writeln!(f, "NT#[ {} ]", self.1[*symbol as usize].as_ref())?;
                    for child in children.iter() {
                        child.pretty(self.1).fmt_indent(f, indent + 1)?;
                    }
                    Ok(())
                }
            }
        }
    }

    /// A parse error.
    #[derive(Debug)]
    pub struct ParseError<A> {
        /// Unexpected current token.
        pub unexpected: Lookahead<A>,
        /// We are currently expecting the following symbols
        pub expecting: Box<[Lookahead<A>]>,
        /// The current parse stack.
        pub parse_stack: Vec<Parse<A>>,
        /// The current state stack.
        pub state_stack: Vec<u32>,
    }

    impl<A: Ord + Clone> super::Table<A> {
        fn parse_error(&self, unexpected: Lookahead<A>,
                       parse_stack: Vec<Parse<A>>,
                       state_stack: Vec<u32>) -> ParseError<A> {
            ParseError {
                unexpected,
                expecting: self.action
                    .equal_range((state_stack.last().unwrap(), ))
                    .map(|(a, _)| a.clone())
                    .collect_vec()
                    .into_boxed_slice(),
                parse_stack,
                state_stack,
            }
        }

        /// Simulate LALR parsing on the given input.
        pub fn simulate_parse<I>(&self, input: I) -> Result<Parse<A>, ParseError<A>>
            where I: IntoIterator<Item=A> {
            let mut state_stack = vec![0];
            let mut parse_stack = Vec::new();
            let mut input = input.into_iter().map(Lookahead::new)
                .chain(std::iter::once(Lookahead::END_OF_INPUT))
                .peekable();
            while let Some(tok) = input.peek() {
                let st = state_stack.last().copied().unwrap();
                let act = match self.action.get((&st, tok)).copied() {
                    Some(act) => act,
                    None => return Err(self.parse_error(input.next().unwrap(), parse_stack, state_stack)),
                };
                match act {
                    Action::Accept => {
                        assert_eq!(parse_stack.len(), 1, "cannot accept more than 1 parse trees");
                        return Ok(parse_stack.into_iter().next().unwrap());
                    }
                    Action::Shift(s) => {
                        let tok = input.next().unwrap();
                        let tok = tok.expect("cannot shift in END_OF_INPUT").to_owned();
                        parse_stack.push(Parse::Terminal(tok));
                        state_stack.push(s);
                    }
                    Action::Reduce { count, symbol } => {
                        let children = parse_stack
                            .split_off(parse_stack.len() - count as usize)
                            .into_boxed_slice();
                        parse_stack.push(Parse::NonTerminal { symbol, children });
                        state_stack.truncate(state_stack.len() - count as usize);
                        let s0 = *state_stack.last().unwrap();
                        let t = *self.goto.get((&s0, &symbol)).unwrap();
                        state_stack.push(t);
                    }
                }
            }
            // Lookahead::END_OF_INPUT is never removed from the input stream
            // therefore it is impossible to exit the 'while' loop normally and reach here
            unreachable!()
        }
    }
}
