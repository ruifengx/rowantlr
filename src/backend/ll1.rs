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

//! `LL(1)` backend for rowantlr, and some common utilities.
//!
//! For each non-terminal `N`:
//! - `DEDUCE_TO_EMPTY(N)` indicates whether `N —→* ε`.
//! - `FIRST(N)` collects all terminal `a` such that `N —→* a β` (`β ∈ V*`).
//!
//! Below is an illustration, one can:
//! - use the [`Grammar`] API to create new grammars.
//! - use [`calc_deduce_to_empty`] to calculate `DEDUCE_TO_EMPTY`.
//! - use [`calc_first`] to calculate `FIRST`.
//!
//! ```
//! #![allow(non_snake_case)]
//! use rowantlr::r#box;
//! use rowantlr::ir::syntax::{Grammar, epsilon, Symbol::*};
//! use rowantlr::backend::ll1::{calc_deduce_to_empty, calc_first};
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
//! // Calculate `DEDUCE_TO_EMPTY` and `FIRST`.
//! let deduce_to_empty = calc_deduce_to_empty(&g);
//! assert_eq!(deduce_to_empty, r#box![false, false, false, true, true, false]);
//! let first = calc_first(&g, &deduce_to_empty);
//! assert_eq!(first, r#box![
//!     r#box!["(", "id"],
//!     r#box!["(", "id"],
//!     r#box!["(", "id"],
//!     r#box!["+"],
//!     r#box!["*"],
//!     r#box!["(", "id"],
//! ]);
//! ```

use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::fmt::{Display, Formatter};
use crate::ir::syntax::{Grammar, Symbol};
use crate::utils::DisplayDot2TeX;

/// A lookahead token.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct Lookahead<A>(Option<A>);

impl<A> Lookahead<A> {
    /// This token indicated the end of the input.
    pub const END_OF_INPUT: Self = Lookahead(None);
    /// A normal token where the input stream continues.
    pub fn new(a: A) -> Self { Lookahead(Some(a)) }
    /// Converts from `&Lookahead<A>` to `Lookahead<&A>`.
    pub fn as_ref(&self) -> Lookahead<&A> { Lookahead(self.0.as_ref()) }
    /// Converts from `&mut Lookahead<A>` to `Lookahead<&mut A>`.
    pub fn as_mut(&mut self) -> Lookahead<&mut A> { Lookahead(self.0.as_mut()) }
    /// Expect the lookahead not being `END_OF_INPUT`.
    pub fn expect(self, msg: &str) -> A { self.0.expect(msg) }
}

impl<A> Lookahead<&'_ A> {
    /// Converts from `Lookahead<&A>` back to an owned `Lookahead<A>`.
    pub fn cloned(self) -> Lookahead<A> where A: Clone { Lookahead(self.0.cloned()) }
}

impl<A> Lookahead<&'_ mut A> {
    /// Converts from `Lookahead<&A>` back to an owned `Lookahead<A>`.
    pub fn cloned(self) -> Lookahead<A> where A: Clone { Lookahead(self.0.cloned()) }
}

impl<A> From<A> for Lookahead<A> {
    fn from(a: A) -> Self { Lookahead::new(a) }
}

impl<A: Display> Display for Lookahead<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            None => write!(f, "<<EOF>>"),
            Some(a) => write!(f, "{}", a),
        }
    }
}

impl<A: DisplayDot2TeX<Env>, Env: ?Sized> DisplayDot2TeX<Env> for Lookahead<A> {
    fn fmt_dot2tex(&self, env: &Env, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            None => write!(f, r"\EOF "),
            Some(a) => write!(f, r"\token{{{}}}", a.display_dot2tex(env)),
        }
    }
}

/// Calculate the `DEDUCE_TO_EMPTY` set for each non-terminal.
///
/// For examples, refer to [module-level documentation](../index.html).
pub fn calc_deduce_to_empty<A>(g: &Grammar<A>) -> Box<[bool]> {
    let mut res = vec![false; g.non_terminals_count()];
    let mut updated = true;
    while updated {
        updated = false;
        for (nt, rules) in g.non_terminals().enumerate() {
            let new_val = rules.iter().any(|expr| expr.iter()
                .all(|x| matches!(x, Symbol::NonTerminal(nt) if res[nt.get()])));
            updated |= res[nt] != new_val;
            res[nt] = new_val;
        }
    }
    res.into_boxed_slice()
}

/// Calculate `FIRST(β)` for any string `β ∈ V*`, according to the provided `FIRST` and
/// `DEDUCE_TO_EMPTY` sets. The `bool` indicates whether or not this string may deduce to empty.
pub fn first_of<'a, A>(expr: impl IntoIterator<Item=&'a Symbol<A>>,
                       first: &'a [Box<[A]>], deduce_to_empty: &[bool]) -> (BTreeSet<A>, bool)
    where A: Clone + Ord + 'a {
    let mut result = BTreeSet::new();
    let nullable = append_first_of::<_, _, _, _, [A]>(
        expr, first, deduce_to_empty, &mut result, &mut false);
    (result, nullable)
}

/// Append `FIRST(β)` to a given [`BTreeSet`] for any string `β ∈ V*`, according to the provided
/// `FIRST` and `DEDUCE_TO_EMPTY` sets. The returned `bool` indicates whether or not this input
/// string may deduce to empty.
pub fn append_first_of<'a, A, R, E, C, I>(expr: E, first: &'a [C], deduce_to_empty: &[bool],
                                          result: &mut BTreeSet<R>, updated: &mut bool) -> bool
    where A: Clone + Into<R> + 'a, R: Ord, I: 'a + ?Sized,
          E: IntoIterator<Item=&'a Symbol<A>>,
          C: Borrow<I>, &'a I: IntoIterator<Item=&'a A> {
    for x in expr {
        match x {
            Symbol::Terminal(t) => {
                *updated |= result.insert(t.clone().into());
                return false;
            }
            Symbol::NonTerminal(nt) => {
                let nt = nt.get();
                for a in first[nt].borrow() {
                    *updated |= result.insert(a.clone().into());
                }
                if !deduce_to_empty[nt] { return false; }
            }
        }
    }
    true
}

/// Calculate the `FIRST` set for each non-terminal.
///
/// For examples, refer to [module-level documentation](../index.html).
pub fn calc_first<A: Ord + Clone>(g: &Grammar<A>, deduce_to_empty: &[bool]) -> Box<[Box<[A]>]> {
    let mut res = vec![BTreeSet::new(); g.non_terminals_count()];
    let mut updated = true;
    while updated {
        updated = false;
        for (nt, expr) in g.rules() {
            let mut cur = std::mem::take(&mut res[nt]);
            append_first_of(expr, &res, deduce_to_empty, &mut cur, &mut updated);
            res[nt] = cur;
        }
    }
    res.into_iter().map(|s| s.into_iter().collect()).collect()
}
