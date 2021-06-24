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
    /// Use a [`NonTerminalIdx`] from `Grammar::add_non_terminal` instead of manually constructing
    /// a [`NonTerminalIdx`]. For test purposes only, use [`NonTerminal`](crate::NonTerminal).
    NonTerminal(NonTerminalIdx),
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
pub type Expr<A> = Box<[Symbol<A>]>;

/// Expression for empty strings (ε).
pub fn epsilon<A>() -> Expr<A> { Box::new([]) as _ }

/// Expressions with carets.
///
/// e.g. to represent RHS of a production rule `A -> a A b` (caret written as dot):
/// - at the very beginning: `. a A b`;
/// - after consuming an `a`: `a . A b`;
///
/// ```
/// # use rowantlr::r#box;
/// # use rowantlr::ir::grammar::{Grammar, CaretExpr, Symbol::*};
/// let _ = Grammar::<()>::build(|g| {
///     let nt = g.add_non_terminal();
///     let expr = r#box![Terminal('a'), NonTerminal(nt), Terminal('b')];
///     let c_expr = CaretExpr::from(&expr);
///     assert_eq!(" . a (NT#1) b", format!("{}", c_expr));
/// });
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
    /// Create a new [`CaretExpr`].
    ///
    /// ```
    /// # use rowantlr::r#box;
    /// # use rowantlr::ir::grammar::{Grammar, CaretExpr, Symbol::*};
    /// let _ = Grammar::<()>::build(|g| {
    ///     let nt = g.add_non_terminal();
    ///     let expr = r#box![Terminal('a'), NonTerminal(nt), Terminal('b')];
    ///     let c_expr = CaretExpr::from(&expr);
    ///     assert_eq!(" . a (NT#1) b", format!("{}", c_expr));
    /// });
    /// ```
    pub fn new(expr: &'a Expr<A>) -> Self { CaretExpr::from(expr) }

    /// Step the caret, return the [`Symbol`] we just step across and a new [`CaretExpr`].
    ///
    /// ```
    /// # use rowantlr::r#box;
    /// # use rowantlr::ir::grammar::{Grammar, CaretExpr, Symbol::*};
    /// let _ = Grammar::<()>::build(|g| {
    ///     let nt = g.add_non_terminal();
    ///     let expr = r#box![Terminal('a'), NonTerminal(nt), Terminal('b')];
    ///     let c_expr = CaretExpr::from(&expr);
    ///     let (_, c_expr) = c_expr.step().unwrap();
    ///     assert_eq!("a . (NT#1) b", format!("{}", c_expr));
    /// });
    /// ```
    pub fn step(self) -> Option<(&'a Symbol<A>, Self)> {
        if self.caret == self.components.len() { return None; }
        Some((&self.components[self.caret], CaretExpr {
            caret: self.caret + 1,
            components: self.components,
        }))
    }

    /// Consumed part: this is usually not useful, exposed for completeness anyways.
    ///
    /// ```
    /// # use rowantlr::r#box;
    /// # use rowantlr::ir::grammar::{Grammar, CaretExpr, Symbol::*};
    /// let _ = Grammar::<()>::build(|g| {
    ///     let nt = g.add_non_terminal();
    ///     let expr = r#box![Terminal('a'), NonTerminal(nt), Terminal('b')];
    ///     let c_expr = CaretExpr::from(&expr);
    ///     let (_, c_expr) = c_expr.step().unwrap();
    ///     assert_eq!(&[Terminal('a')], c_expr.consumed_part());
    /// });
    /// ```
    pub fn consumed_part(&self) -> &[Symbol<A>] { &self.components[..self.caret] }

    /// Symbols yet to consume.
    ///
    /// ```
    /// # use rowantlr::r#box;
    /// # use rowantlr::ir::grammar::{Grammar, CaretExpr, Symbol::*};
    /// let _ = Grammar::<()>::build(|g| {
    ///     let nt = g.add_non_terminal();
    ///     let expr = r#box![Terminal('a'), NonTerminal(nt), Terminal('b')];
    ///     let c_expr = CaretExpr::from(&expr);
    ///     let (_, c_expr) = c_expr.step().unwrap();
    ///     assert_eq!(&[NonTerminal(nt), Terminal('b')], c_expr.rest_part());
    /// });
    /// ```
    pub fn rest_part(&self) -> &[Symbol<A>] { &self.components[self.caret..] }
}

/// Wrapped non-terminal, subscript into [`Grammar`]s.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct NonTerminalIdx(NonZeroUsize);

impl Display for NonTerminalIdx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(NT#{})", self.0)
    }
}

impl NonTerminalIdx {
    pub(crate) fn get(self) -> usize { self.0.get() }
}

/// Builder for [`Grammar`]s.
#[derive(Debug, Eq, PartialEq)]
pub struct GrammarBuilder<A> {
    pub(crate) rules: Vec<Vec<Expr<A>>>,
}

impl<A> Default for GrammarBuilder<A> {
    fn default() -> Self {
        GrammarBuilder { rules: vec![Vec::new()] }
    }
}

impl<A> GrammarBuilder<A> {
    /// Create a new grammar, same as `Grammar::default`.
    pub fn new() -> Self { Default::default() }

    /// Add a new non-terminal.
    pub fn add_non_terminal(&mut self) -> NonTerminalIdx {
        let nt = self.rules.len();
        self.rules.push(Vec::new());
        NonTerminalIdx(NonZeroUsize::new(nt).unwrap())
    }

    /// Add many new non-terminals all at once.
    pub fn add_non_terminals<const N: usize>(&mut self) -> [NonTerminalIdx; N] {
        let mut res = [NonTerminalIdx(NonZeroUsize::new(42).unwrap()); N];
        for i in 0..N {
            res[i] = self.add_non_terminal();
        }
        res
    }

    /// Add a new production rule to a non-terminal.
    pub fn add_rule(&mut self, nt: NonTerminalIdx, rule: Expr<A>) {
        self.rules[nt.get()].push(rule)
    }

    /// Mark a non-terminal symbol as a start symbol.
    pub fn mark_as_start(&mut self, nt: NonTerminalIdx) {
        let nt = Symbol::NonTerminal(nt);
        self.rules[0].push(Box::new([nt]) as _)
    }

    /// Finish building, and get the final [`Grammar`].
    pub fn finish(self) -> Grammar<A> {
        let mut all_rules = Vec::new();
        let mut indices = Vec::new();
        for mut rules in self.rules {
            indices.push(all_rules.len());
            all_rules.append(&mut rules);
        }
        Grammar {
            all_rules: all_rules.into_boxed_slice(),
            indices: indices.into_boxed_slice(),
        }
    }
}

/// (Augmented) Grammars, i.e. set of production rules, with a start symbol `S` (indexed 0) not
/// appearing in RHS of any production rules.
pub struct Grammar<A> {
    all_rules: Box<[Expr<A>]>,
    indices: Box<[usize]>,
}

impl<A> Grammar<A> {
    /// Convenient method for building a grammar.
    pub fn build(proc: impl for<'a> FnOnce(&'a mut GrammarBuilder<A>)) -> Self {
        let mut builder = GrammarBuilder::new();
        proc(&mut builder);
        builder.finish()
    }

    /// Number of non-terminals in this grammar.
    pub fn non_terminals_count(&self) -> usize { self.indices.len() }

    /// Number of rules in this grammar.
    pub fn rules_count(&self) -> usize { self.all_rules.len() }

    /// Iterate over all the production rules in the grammar.
    pub fn rules(&self) -> impl Iterator<Item=(usize, &[Symbol<A>])> + '_ {
        self.non_terminals().enumerate()
            .flat_map(|(n, rs)| rs.iter().map(move |e| (n, &e[..])))
    }

    /// Iterate over all the production rules grouped by non-terminals.
    pub fn non_terminals(&self) -> impl Iterator<Item=&[Expr<A>]> + '_ {
        self.indices.iter().copied()
            .zip(self.indices.iter().copied().dropping(1)
                .chain(std::iter::once(self.all_rules.len())))
            .map(move |(l, r)| &self.all_rules[l..r])
    }
}
