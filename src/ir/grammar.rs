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

//! IR for grammars: sets of production rules.

use std::fmt::{self, Display, Formatter};
use std::num::NonZeroUsize;
use derivative::Derivative;
use itertools::Itertools;

/// Terminal and non-terminal symbols.
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub enum Symbol<A> {
    /// Terminals are closed terms.
    Terminal(A),
    /// Non-terminals are subscripts into [`Grammar`]s.
    ///
    /// Index 0 reserved for the special start symbol `S`, which is not allowed to be referenced
    /// in RHS of any production rules, and thus here ruled out as a non-terminal symbol.
    ///
    /// Use [`nonzero`] to for `NonZeroUsize` literals.
    NonTerminal(NonZeroUsize),
}

impl<A: Display> Display for Symbol<A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Symbol::Terminal(a) => write!(f, "{}", a),
            Symbol::NonTerminal(n) => write!(f, "{}", *n),
        }
    }
}

/// Expressions are sequences of [`Symbol::Terminal`]s and [`Symbol::NonTerminal`]s.
pub type Expr<A> = Vec<Symbol<A>>;

/// Expressions with carets.
///
/// e.g. to represent RHS of a production rule `A -> a A b` (caret written as dot):
/// - at the very beginning: `. a A b`;
/// - after consuming an `a`: `a . A b`;
///
/// ```
/// use std::num::NonZeroUsize;
/// # use rowantlr::ir::grammar::CaretExpr;
/// # use rowantlr::ir::grammar::Symbol::*;
/// # use rowantlr::nonzero;
/// let expr = vec![Terminal('a'), NonTerminal(nonzero!(1)), Terminal('b')];
/// let c_expr = CaretExpr::from(&expr);
/// assert_eq!(" . a 1 b", format!("{}", c_expr));
/// ```
#[derive(Debug, Clone, Copy)]
#[derive(Derivative)]
#[derivative(PartialEq, Eq)]
pub struct CaretExpr<'a, A> {
    caret: usize,
    #[derivative(PartialEq(compare_with = "std::ptr::eq"))]
    components: &'a Expr<A>,
}

impl<'a, A> From<&'a Expr<A>> for CaretExpr<'a, A> {
    fn from(expr: &Expr<A>) -> CaretExpr<'_, A> {
        CaretExpr { caret: 0, components: expr }
    }
}

impl<'a, A: Display> Display for CaretExpr<'a, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} . {}",
               self.components[..self.caret].iter().format(" "),
               self.components[self.caret..].iter().format(" "))
    }
}

impl<'a, A> CaretExpr<'a, A> {
    /// Step the caret, return the [`Symbol`] we just step across and a new [`CaretExpr`].
    ///
    /// ```
    /// use std::num::NonZeroUsize;
    /// # use rowantlr::ir::grammar::CaretExpr;
    /// # use rowantlr::ir::grammar::Symbol::*;
    /// # use rowantlr::nonzero;
    /// let expr = vec![Terminal('a'), NonTerminal(nonzero!(1)), Terminal('b')];
    /// let c_expr = CaretExpr::from(&expr);
    /// let (_, c_expr) = c_expr.step().unwrap();
    /// assert_eq!("a . 1 b", format!("{}", c_expr));
    /// ```
    pub fn step(self) -> Option<(&'a Symbol<A>, Self)> {
        if self.caret == self.components.len() { return None; }
        Some((&self.components[self.caret], CaretExpr {
            caret: self.caret + 1,
            components: self.components,
        }))
    }
}

/// (Augmented) Grammars, i.e. set of production rules, with a start symbol `S` (indexed 0) not
/// appearing in RHS of any production rules.
#[derive(Debug, Eq, PartialEq)]
pub struct Grammar<A> {
    pub(crate) rules: Vec<Vec<Expr<A>>>,
}

/// Wrapped non-terminal, subscript into [`Grammar`]s.
pub struct NonTerminalIdx(usize);

impl<A> Default for Grammar<A> {
    fn default() -> Self {
        Grammar { rules: vec![Vec::new()] }
    }
}

impl<A> Grammar<A> {
    /// Create a new grammar, same as `Grammar::default`.
    pub fn new() -> Self { Default::default() }

    /// Add a new non-terminal.
    pub fn add_non_terminal(&mut self) -> NonTerminalIdx {
        let nt = self.rules.len();
        self.rules.push(Vec::new());
        NonTerminalIdx(nt)
    }

    /// Add a new production rule to a non-terminal.
    pub fn add_rule(&mut self, nt: NonTerminalIdx, rule: Expr<A>) {
        self.rules[nt.0].push(rule)
    }

    /// Mark a non-terminal symbol as a start symbol.
    pub fn mark_as_start(&mut self, nt: NonTerminalIdx) {
        let nt = Symbol::NonTerminal(NonZeroUsize::new(nt.0).unwrap());
        self.rules[0].push(vec![nt])
    }
}
