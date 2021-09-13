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
//! use rowantlr::ir::grammar::{Grammar, epsilon, Symbol::*};
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

use std::collections::BTreeSet;
use itertools::Itertools;
use itertools::FoldWhile::*;

use crate::ir::grammar::{Grammar, Symbol};
use crate::utils::continue_if_with;

/// A lookahead token.
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
            let new_val = rules.iter().any(|expr| {
                !expr.iter().fold_while((), |(), x| match x {
                    Symbol::Terminal(_) => Done(()),
                    Symbol::NonTerminal(nt) =>
                        continue_if_with(res[nt.get()], ()),
                }).is_done()
            });
            updated |= res[nt] != new_val;
            res[nt] = new_val;
        }
    }
    res.into_boxed_slice()
}

/// Calculate `FIRST(β)` for any string `β ∈ V*`, according to the provided `FIRST` and
/// `DEDUCE_TO_EMPTY` sets.
pub fn first_of<'a, A, C>(expr: impl IntoIterator<Item=&'a Symbol<A>>,
                          first: &'a [C], deduce_to_empty: &[bool]) -> BTreeSet<A>
    where A: Ord + Clone + 'a, &'a C: IntoIterator<Item=&'a A> {
    let mut result = BTreeSet::new();
    append_first_of(expr, first, deduce_to_empty, &mut result);
    result
}

/// Append `FIRST(β)` to a given [`BTreeSet`] for any string `β ∈ V*`, according to the provided
/// `FIRST` and `DEDUCE_TO_EMPTY` sets.
pub fn append_first_of<'a, A, C>(expr: impl IntoIterator<Item=&'a Symbol<A>>,
                                 first: &'a [C], deduce_to_empty: &[bool],
                                 result: &mut BTreeSet<A>) -> bool
    where A: Ord + Clone + 'a, &'a C: IntoIterator<Item=&'a A> {
    let mut updated = false;
    for x in expr {
        match x {
            Symbol::Terminal(t) => {
                updated |= result.insert(t.clone());
                break;
            }
            Symbol::NonTerminal(nt) => {
                let nt = nt.get();
                for a in &first[nt] {
                    updated |= result.insert(a.clone());
                }
                if !deduce_to_empty[nt] { break; }
            }
        }
    }
    updated
}

/// Calculate the `FIRST` set for each non-terminal.
///
/// For examples, refer to [module-level documentation](../index.html).
pub fn calc_first<A: Ord + Clone>(g: &Grammar<A>, deduce_to_empty: &[bool]) -> Box<[Box<[A]>]> {
    let mut res = vec![BTreeSet::new(); g.non_terminals_count()];
    let mut updated = true;
    while updated {
        for (nt, expr) in g.rules() {
            let mut cur = std::mem::take(&mut res[nt]);
            updated = append_first_of(expr, &res, deduce_to_empty, &mut cur);
            res[nt] = cur;
        }
    }
    res.into_iter().map(|s| s.into_iter().collect()).collect()
}
