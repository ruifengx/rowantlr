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

//! IR for lexical specifications.
//!
//! - Use the [`Expr`] API to construct regular expressions;
//! - Use [`Expr::build`], [`dfa::BuildResult::try_resolve`] to build DFAs;
//! - Run a DFA on some input with [`Dfa::run`] and [`dfa::Resolved::run`];
//!
//! Below is an example:
//!
//! ```
//! use rowantlr::ir::lexical::{Expr, dfa::InvalidInput};
//! let expr = Expr::concat([
//!     Expr::singleton('a'),
//!     Expr::many(Expr::union([
//!         Expr::singleton('a'),
//!         Expr::singleton('b'),
//!     ])),
//!     Expr::singleton('b'),
//! ]);
//! let resolved = expr.build().try_resolve().unwrap();
//! assert_eq!(Some(()), resolved.run("aab".chars()).unwrap());
//! assert_eq!(Some(()), resolved.run("ab".chars()).unwrap());
//! assert_eq!(Some(()), resolved.run("abab".chars()).unwrap());
//! assert_eq!(None, resolved.run("aa".chars()).unwrap());
//! let InvalidInput { current_state, current_input, remaining_input }
//!     = resolved.run("baaa".chars()).unwrap_err();
//! assert_eq!(current_state, 0);
//! assert_eq!(current_input, 'b');
//! assert_eq!(remaining_input.as_str(), "aaa");
//! ```

use std::collections::BTreeSet;
use std::ops::{BitAnd, BitOr, Deref, DerefMut};
use crate::utils::{Dict, IterHelper};

/// Regular expressions.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Expr<A>(Op<A, Expr<A>>);

impl<A> Deref for Expr<A> {
    type Target = Op<A, Expr<A>>;
    fn deref(&self) -> &Self::Target { &self.0 }
}

impl<A> DerefMut for Expr<A> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

/// Regular expression (or regular language) operations.
#[derive(Debug, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum Op<A, R> {
    /// `{ a }` is a regular language.
    Singleton(A),
    /// Union.
    Union(Vec<R>),
    /// Concatenation.
    Concat(Vec<R>),
    /// Positive closure (`+`).
    Some(Box<R>),
}

impl<A> Expr<A> {
    /// Singleton: `a`.
    pub const fn singleton(a: A) -> Expr<A> { Expr(Op::Singleton(a)) }
    /// Empty string: `ε`.
    pub fn epsilon() -> Expr<A> { Expr::concat([]) }
    /// Union: `x₁ | x₂ | ... | xₙ`.
    pub fn union(xs: impl IntoIterator<Item=Expr<A>>) -> Expr<A> {
        let mut has_epsilon = false;
        let mut result = Vec::new();
        for x in xs {
            match x.0 {
                Op::Union(xs) => result.extend(xs.into_iter()),
                x => {
                    // ε | x = x  iff x =>* ε
                    if !(matches!(x, Op::Concat(ref xs) if xs.is_empty()) && has_epsilon) {
                        has_epsilon = true;
                        result.push(Expr(x))
                    }
                }
            }
        }
        Expr(Op::Union(result))
    }
    /// Concatenation: `x₁ x₂ ... xₙ`.
    pub fn concat(xs: impl IntoIterator<Item=Expr<A>>) -> Expr<A> {
        let mut result = Vec::new();
        for x in xs {
            match x.0 {
                Op::Concat(xs) => result.extend(xs.into_iter()),
                x => result.push(Expr(x)),
            }
        }
        match result.len() {
            1 => result.into_iter().next().unwrap(),
            _ => Expr(Op::Concat(result)),
        }
    }
    /// Positive closure: `x+`.
    pub fn some(expr: Expr<A>) -> Expr<A> {
        Expr(match expr.0 {
            Op::Concat(xs) if xs.is_empty() => Op::Concat(xs), // ε+ = ε
            x @ Op::Some(_) => x, // (x+)+ = x+
            x => Op::Some(Box::new(Expr(x))),
        })
    }
    /// Optional: `x? = ε | x`.
    pub fn optional(expr: Expr<A>) -> Expr<A> {
        Expr::union([Expr::epsilon(), expr])
    }
    /// Closure, or the Kleene star: `x* = ε | x+`.
    pub fn many(expr: Expr<A>) -> Expr<A> {
        Expr::optional(Expr::some(expr))
    }
}

/// Information of a (sub-) regular expression.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct ExprInfo {
    /// Whether or not `e =>* ε`.
    pub nullable: bool,
    /// `firstpos(e)`: the set of positions in `e` which correspond to some symbol `a` such that
    /// `e =>* aβ`, i.e. the symbols appears in the first places of possible sentences in `L(e)`.
    pub first_pos: BTreeSet<usize>,
    /// `lastpos(e)`: the set of positions in `e` which correspond to some symbol `a` such that
    /// `e =>* βa`, i.e. the symbols appears in the last places of possible sentences in `L(e)`.
    pub last_pos: BTreeSet<usize>,
}

impl ExprInfo {
    /// Information for `ε` nodes.
    pub fn epsilon() -> ExprInfo {
        ExprInfo {
            nullable: true,
            ..ExprInfo::default()
        }
    }

    /// Information for leaf nodes.
    pub fn singleton(pos: usize) -> ExprInfo {
        let pos = std::iter::once(pos).collect::<BTreeSet<_>>();
        ExprInfo {
            nullable: false,
            first_pos: pos.clone(),
            last_pos: pos,
        }
    }
}

impl BitOr for ExprInfo {
    type Output = ExprInfo;
    fn bitor(self, rhs: Self) -> Self::Output {
        ExprInfo {
            nullable: self.nullable || rhs.nullable,
            first_pos: self.first_pos.union(&rhs.first_pos).copied().collect(),
            last_pos: self.last_pos.union(&rhs.last_pos).copied().collect(),
        }
    }
}

impl BitAnd for ExprInfo {
    type Output = ExprInfo;
    fn bitand(self, rhs: Self) -> Self::Output {
        ExprInfo {
            nullable: self.nullable && rhs.nullable,
            first_pos: if self.nullable {
                self.first_pos.union(&rhs.first_pos).copied().collect()
            } else {
                self.first_pos
            },
            last_pos: if rhs.nullable {
                self.last_pos.union(&rhs.last_pos).copied().collect()
            } else {
                rhs.last_pos
            },
        }
    }
}

/// How is this position generated.
#[derive(Debug)]
pub enum PosInfo<A, Tag> {
    /// From a non-`ε` leaf node, i.e. an [`Op::Singleton`].
    Normal(A),
    /// From a phantom "accept" node, with a tag attached.
    Accept(Tag),
}

/// Recursive visitor for [`Expr`]s.
pub trait ExprVisitor<A, Tag> {
    /// Visit a `or-node`, part of an [`Op::Union`].
    fn visit_union(&mut self, lhs: &ExprInfo, rhs: &ExprInfo);
    /// Visit a `cat-node`, part of an [`Op::Concat`].
    fn visit_cat(&mut self, lhs: &ExprInfo, rhs: &ExprInfo);
    /// Visit a `some-node`, i.e. an [`Op::Some`].
    fn visit_some(&mut self, x: &ExprInfo);
    /// Visit a leaf node, i.e. an [`Op::Singleton`].
    fn gen_info(&mut self, a: PosInfo<A, Tag>) -> ExprInfo;

    /// Visit a leaf node, i.e. an [`Op::Singleton`].
    fn gen_singleton_info(&mut self, a: A) -> ExprInfo {
        self.gen_info(PosInfo::Normal(a))
    }
    /// Visit a leaf node, i.e. an [`Op::Singleton`].
    fn gen_accept_info(&mut self, t: Tag) -> ExprInfo {
        self.gen_info(PosInfo::Accept(t))
    }
    /// Visit a `cat-node`, part of an [`Op::Concat`].
    /// More control on how [`ExprInfo`]s are merged.
    fn map_cat(&mut self, lhs: ExprInfo, rhs: ExprInfo) -> ExprInfo {
        self.visit_cat(&lhs, &rhs);
        lhs & rhs
    }
    /// Visit a `or-node`, part of an [`Op::Union`].
    /// More control on how [`ExprInfo`]s are merged.
    fn map_union(&mut self, lhs: ExprInfo, rhs: ExprInfo) -> ExprInfo {
        self.visit_union(&lhs, &rhs);
        lhs | rhs
    }
    /// Visit a `some-node`, i.e. an [`Op::Some`].
    /// More control on how the [`ExprInfo`] is transformed.
    fn map_some(&mut self, x: ExprInfo) -> ExprInfo {
        self.visit_some(&x);
        x
    }
}

impl<A: Clone> Expr<A> {
    fn traverse<Tag>(visitor: &mut impl ExprVisitor<A, Tag>, expr: &Expr<A>) -> ExprInfo {
        match expr.deref() {
            Op::Singleton(a) => visitor.gen_singleton_info(a.clone()),
            Op::Union(xs) => xs.iter()
                .reduce_map(visitor, Expr::traverse, ExprVisitor::map_union)
                .unwrap_or_default(),
            Op::Concat(xs) => xs.iter()
                .reduce_map(visitor, Expr::traverse, ExprVisitor::map_cat)
                .unwrap_or_else(|| ExprInfo { nullable: true, ..ExprInfo::default() }),
            Op::Some(x) => {
                let x = Expr::traverse(visitor, x);
                visitor.map_some(x)
            }
        }
    }

    /// Traverse the expression tree, building up [`ExprInfo`] on the fly, and use the
    /// information during the traversal.
    pub fn traverse_with_info<Tag>(&self, visitor: &mut impl ExprVisitor<A, Tag>) -> ExprInfo {
        Expr::traverse(visitor, self)
    }

    /// For regular expression `e`, collect the information as if we are traversing the extended
    /// regular expression `e#` (where `# ∉ Σ`).
    pub fn traverse_extended<Tag>(&self, visitor: &mut impl ExprVisitor<A, Tag>, tag: Tag) -> ExprInfo {
        let main = self.traverse_with_info(visitor);
        let acc = visitor.gen_accept_info(tag);
        visitor.map_cat(main, acc)
    }
}

/// Definite Finite State Automata.
#[derive(Debug)]
pub struct Dfa<A> {
    /// Number of states in this DFA.
    pub state_count: usize,
    /// Transitions (arcs) of this DFA.
    pub transitions: Dict<(usize, A, usize)>,
}

/// DFA related data structures.
pub mod dfa {
    use std::borrow::Borrow;
    use super::{Dfa, ExprInfo, ExprVisitor, PosInfo};

    use std::rc::Rc;
    use std::collections::{BTreeSet, BTreeMap, VecDeque};
    use itertools::Itertools;
    use derivative::Derivative;
    use crate::utils::Dict;

    /// Invalid input for some [`Dfa`].
    ///
    /// Since our transition function of a [`Dfa`] is partial, it is possible that running on some
    /// prefix of an input string is undefined.
    #[derive(Debug, Eq, PartialEq)]
    pub struct InvalidInput<A, I> {
        /// The last state this [`Dfa`] reaches.
        pub current_state: usize,
        /// The input character for which the transition function is not defined.
        pub current_input: A,
        /// Possible remaining input.
        pub remaining_input: I,
    }

    impl<A: Ord> Dfa<A> {
        /// Run this DFA on the specific `input`, starting from some `start_state`.
        /// If no transition is available, return a tuple of state, input char, and rest input.
        pub fn run<'a, C, I>(&self, start_state: usize, input: I)
                             -> Result<usize, InvalidInput<C, I::IntoIter>>
            where A: 'a, C: 'a + Borrow<A>, I: IntoIterator<Item=C> {
            let mut remaining_input = input.into_iter();
            let mut current_state = start_state;
            for current_input in &mut remaining_input {
                current_state = match self.transitions
                    .get((&current_state, current_input.borrow())) {
                    Some(p) => *p,
                    None => return Err(InvalidInput {
                        current_state,
                        current_input,
                        remaining_input,
                    }),
                };
            }
            Ok(current_state)
        }
    }

    /// Calculate `followpos(p)` for every position `p` in a regular expression `e`.
    /// This piece of information can be used to construct a DFA.
    #[derive(Derivative)]
    #[derivative(Default(bound = ""))]
    pub struct Builder<A, Tag>(Vec<PosEntry<A, Tag>>);

    /// Information related to a position in a regular expression.
    #[derive(Debug)]
    pub struct PosEntry<A, Tag> {
        /// Whether this state is an accept state.
        pub info: PosInfo<A, Tag>,
        /// The `followpos(p)` set for this position `p`.
        pub follow_pos: BTreeSet<usize>,
    }

    impl<A, Tag> PosEntry<A, Tag> {
        /// Returns `true` if `info` is a `PosInfo::Accept`.
        pub fn is_accept(&self) -> bool {
            matches!(&self.info, PosInfo::Accept(_))
        }
    }

    impl<A, Tag> Builder<A, Tag> {
        /// Create a new [`Builder`], the same as [`Default::default`].
        pub fn new() -> Self { Builder::default() }
        /// Finish by getting the collected information. See also [`Builder::build`].
        pub fn finish(self) -> Vec<PosEntry<A, Tag>> { self.0 }
    }

    impl<A, Tag> ExprVisitor<A, Tag> for Builder<A, Tag> {
        fn visit_union(&mut self, _: &ExprInfo, _: &ExprInfo) {}
        fn visit_cat(&mut self, lhs: &ExprInfo, rhs: &ExprInfo) {
            for i in lhs.last_pos.iter().copied() {
                self.0[i].follow_pos.extend(rhs.first_pos.iter().copied())
            }
        }
        fn visit_some(&mut self, x: &ExprInfo) {
            self.visit_cat(x, x)
        }
        fn gen_info(&mut self, p_info: PosInfo<A, Tag>) -> ExprInfo {
            let info = ExprInfo::singleton(self.0.len());
            self.0.push(PosEntry { info: p_info, follow_pos: BTreeSet::new() });
            info
        }
    }

    impl<A: Ord + Clone, Tag> Builder<A, Tag> {
        /// Build a [`Dfa`].
        ///
        /// To preserve as much information as possible, the map from DFA states and positions in
        /// regular expressions is provided as is, without substitution with tags in [`PosEntry`]s.
        /// In this form, conflicts can be easily detected, and error reports can be associated with
        /// the original regular expressions.
        pub fn build(self, expr_info: ExprInfo) -> BuildResult<A, Tag> {
            let start = Rc::new(expr_info.first_pos);
            // state -> state index
            let mut state_mapping = BTreeMap::new();
            state_mapping.insert(start.clone(), 0);
            // state index * input -> state index
            let mut state_transition = BTreeMap::new();
            // [state * state index]
            let mut queue = VecDeque::new();
            queue.push_back((start, 0));

            while let Some((q0, k0)) = queue.pop_front() { // q0: state
                debug_assert_eq!(state_mapping.get(&q0), Some(&k0));
                let mut trans = BTreeMap::<A, BTreeSet<usize>>::new();
                for p in q0.iter().copied() { // p: position
                    if let PosInfo::Normal(a) = &self.0[p].info {
                        trans.entry(a.clone()).or_default().extend(&self.0[p].follow_pos);
                    }
                }
                for (a, q) in trans {
                    let q = Rc::new(q);
                    let k = state_mapping.len();
                    let k = *state_mapping.entry(q.clone()).or_insert_with(|| {
                        queue.push_back((q.clone(), k));
                        k
                    });
                    state_transition.insert((k0, a), k);
                }
            }

            let state_mapping = {
                let mut result = vec![BTreeSet::new(); state_mapping.len()];
                for (q, k) in state_mapping {
                    result[k] = Rc::try_unwrap(q).unwrap();
                }
                result
            };

            BuildResult {
                dfa: Dfa {
                    state_count: state_mapping.len(),
                    transitions: state_transition.into_iter()
                        .map(|((s, a), t)| (s, a, t)).collect(),
                },
                state_mapping,
                position_info: self.0,
            }
        }
    }

    /// Information for reachability for states in a DFA for a specific [`BuildResult`].
    pub struct Reachability<A> {
        /// The mapping is `state ->? (state, position)`:
        /// - `state` for the predecessor state;
        /// - `position` for the position in the original regular expression.
        ///
        /// If `predecessor[s] == Some((t, a))`, we have in turn:
        /// - there is an edge (labelled `a`) from `t` to `s`;
        /// and for each position `q` in `s`, we have such a position `p` that
        /// - `p` is in `state_mapping[t]`;
        /// - `position_info[p] = Normal(a)`;
        /// - `q` is in `followpos(p)`;
        /// The existence of `p` follows immediately from the existence of the edge.
        ///
        /// At most one predecessor is recorded here. If the [`BuildResult`] is obtained by
        /// [`built`](Builder::build) from an [`Expr`](super::Expr), and if this reachability
        /// information is generated from [`BuildResult::reachability`], follow this predecessor
        /// should give a shortest path from the initial state to the current state.
        pub predecessor: Dict<(usize, (usize, A))>,
    }

    impl<A: Clone> Reachability<A> {
        /// Try to generate a path from the initial state to a specific state.
        /// Every element in the path is a `(state, input)` pair.
        pub fn try_get_path_for(&self, state: usize) -> Option<Vec<(usize, A)>> {
            let mut current_state = state;
            let mut result_path = Vec::new();
            while let Some((pred, a)) = self.predecessor.get((&current_state, )) {
                result_path.push((*pred, a.clone()));
                current_state = *pred;
            }
            if current_state != 0 { return None; }
            result_path.reverse();
            Some(result_path)
        }
    }

    /// Result from [`build`](Builder::build)ing a [`Dfa`].
    #[derive(Debug)]
    pub struct BuildResult<A, Tag> {
        /// The resulted DFA.
        pub dfa: Dfa<A>,
        /// Mapping from DFA states to positions in the original regular expression.
        pub state_mapping: Vec<BTreeSet<usize>>,
        /// Mapping from [`PosInfo::Accept`] positions to their tags.
        pub position_info: Vec<PosEntry<A, Tag>>,
    }

    impl<A: Ord + Clone, Tag> BuildResult<A, Tag> {
        /// Calculate reachability information for DFA states.
        pub fn reachability(&self) -> Reachability<A> {
            let mut predecessor = vec![None; self.dfa.state_count];
            for (s, a, t) in self.dfa.transitions.iter() {
                if s == t || predecessor[*t].is_some() { continue; }
                predecessor[*t] = Some((*s, a.clone()));
            }
            let predecessor = predecessor.into_iter().enumerate()
                .filter_map(|(k, p)| Some((k, p?))).collect();
            Reachability { predecessor }
        }
    }

    impl<A: PartialEq, Tag> BuildResult<A, Tag> {
        /// Resolve a path in the [`Dfa`] to a series of positions.
        pub fn path_to_positions(&self, path: &[(usize, A)]) -> Vec<usize> {
            let mut path = path.iter().rev();
            let mut result = Vec::new();
            let mut last_pos = match path.next() {
                None => return result,
                Some((state, input)) =>
                    self.state_mapping[*state].iter().copied()
                        .find(|&p| matches!(&self.position_info[p].info,
                            PosInfo::Normal(a) if a == input))
                        .unwrap(),
            };
            result.push(last_pos);
            for (state, input) in path {
                last_pos = self.state_mapping[*state].iter().copied()
                    .find(|&p| self.position_info[p].follow_pos.contains(&last_pos)
                        && matches!(&self.position_info[p].info, PosInfo::Normal(a) if a == input))
                    .unwrap();
                result.push(last_pos);
            }
            result.reverse();
            result
        }
    }

    /// [`BuildResult`] with conflicts resolved.
    #[derive(Debug)]
    pub struct Resolved<A, Tag> {
        /// The resulted DFA.
        pub dfa: Dfa<A>,
        /// Mapping from states to tags.
        pub tags: Dict<(usize, Tag)>,
    }

    impl<A: Ord, Tag: Clone> Resolved<A, Tag> {
        /// Run the resolved DFA on this specific input and get the result.
        pub fn run<'a, C, I>(&self, input: I) -> Result<Option<Tag>, InvalidInput<C, I::IntoIter>>
            where A: 'a, C: 'a + Borrow<A>, I: IntoIterator<Item=C> {
            let s = self.dfa.run(0, input)?;
            Ok(self.tags.get((&s, )).map(Tag::clone))
        }
    }

    /// Conflict in DFA states.
    #[derive(Debug)]
    pub struct Conflict {
        /// An example input string for this conflict. Characters of this input string is encoded
        /// as positions on the original regular expression (instead of the input character `A`)
        /// for better error reporting.
        pub example_input: Box<[usize]>,
        /// Possible tags for `example_input`, encoded as positions on the original regular expression.
        pub conflicting_tags: Box<[usize]>,
    }

    /// Result for resolving conflicts.
    pub enum ResolveError<A, Tag, Input> {
        /// No error. Original mapping returned.
        Resolved(Box<[usize]>),
        /// There is only one tag for the input string. Thus there is no conflict at all.
        NoConflict(usize),
        /// The state for the input string is not an accepted state. This input was refused.
        NotAcceptState(NotAcceptState<Tag>),
        /// The specified input is not defined for this DFA.
        InvalidInput(InvalidInput<A, Input>),
    }

    /// Decide what to do next for a non-accepting state.
    pub struct NotAcceptState<Tag> {
        /// The index of the state in the [`Dfa`].
        pub state_index: usize,
        /// The tag intended for that state.
        pub tag_intended: Tag,
    }

    impl<A: Ord, Tag: Ord> BuildResult<A, Tag> {
        /// Try to resolve a conflict using a hint: which `tag` should be assigned to some `input`.
        pub fn apply_hint<'a, I>(&mut self, input: I, tag_intended: Tag)
                                 -> ResolveError<&'a A, Tag, I::IntoIter>
            where I: IntoIterator<Item=&'a A>, A: 'a {
            let state_index = match self.dfa.run(0, input) {
                Err(err) => return ResolveError::InvalidInput(err),
                Ok(s) => s,
            };
            let ps = self.state_mapping[state_index].iter().copied()
                .filter(|&p| self.position_info[p].is_accept())
                .collect_vec();
            match ps.len() {
                0 => ResolveError::NotAcceptState(NotAcceptState { state_index, tag_intended }),
                1 => ResolveError::NoConflict(ps[0]),
                _ => {
                    let k = self.position_info.len();
                    self.position_info.push(PosEntry {
                        info: PosInfo::Accept(tag_intended),
                        follow_pos: BTreeSet::default(),
                    });
                    self.state_mapping[state_index] = std::iter::once(k).collect();
                    ResolveError::Resolved(ps.into_boxed_slice())
                }
            }
        }

        /// Try to resolve conflicts using a series of hints.
        /// See also [`apply_hint`](BuildResult::apply_hint).
        pub fn apply_hints<'a, I, H>(&mut self, hints: H) -> Vec<ResolveError<&'a A, Tag, I::IntoIter>>
            where I: IntoIterator<Item=&'a A>, H: IntoIterator<Item=(I, Tag)>, A: 'a {
            hints.into_iter().map(|hint| self.apply_hint(hint.0, hint.1)).collect()
        }

        /// Resolve all the conflicts and produce the final DFA.
        pub fn try_resolve(self) -> Result<Resolved<A, Tag>, (Self, Vec<Conflict>)>
            where A: Clone, Tag: Clone {
            let mut result = BTreeMap::new();
            let mut errors = Vec::new();
            let mut reachability = None;
            for (k, ps) in self.state_mapping.iter().enumerate() {
                let ps = ps.iter().copied()
                    .filter(|&p| self.position_info[p].is_accept())
                    .collect_vec();
                match ps.len() {
                    0 => {}
                    1 => match &self.position_info[ps[0]].info {
                        PosInfo::Accept(tag) => { result.insert(k, tag.clone()); }
                        PosInfo::Normal(_) => unreachable!(),
                    }
                    _ => {
                        let reach = reachability.get_or_insert_with(|| self.reachability());
                        let path = reach.try_get_path_for(k).unwrap();
                        errors.push(Conflict {
                            example_input: self.path_to_positions(&path).into_boxed_slice(),
                            conflicting_tags: ps.into_boxed_slice(),
                        })
                    }
                }
            }
            if errors.is_empty() {
                Ok(Resolved {
                    dfa: self.dfa,
                    tags: result.into_iter().collect(),
                })
            } else {
                Err((self, errors))
            }
        }
    }
}

impl<A: Ord + Clone> Expr<A> {
    /// Build a [`Dfa`] from a regular expression.
    /// See also [`dfa::Builder::build`].
    pub fn build(&self) -> dfa::BuildResult<A, ()> {
        let mut builder = dfa::Builder::default();
        let info = self.traverse_extended(&mut builder, ());
        builder.build(info)
    }
}
