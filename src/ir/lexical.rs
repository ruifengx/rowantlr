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

//! IR for lexical specifications.

use std::ops::{Deref, DerefMut};

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
    /// Empty string.
    Epsilon,
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
    /// Empty string: `ε`.
    pub const EPSILON: Expr<A> = Expr(Op::Epsilon);
    /// Singleton: `a`.
    pub const fn singleton(a: A) -> Expr<A> { Expr(Op::Singleton(a)) }
    /// Union: `x₁ | x₂ | ... | xₙ`.
    pub fn union(xs: impl IntoIterator<Item=Expr<A>>) -> Expr<A> {
        let mut has_epsilon = false;
        let mut result = Vec::new();
        for x in xs {
            match x.0 {
                Op::Union(xs) => result.extend(xs.into_iter()),
                x => {
                    // ε | x = x  iff x =>* ε
                    if !(matches!(x, Op::Epsilon) && has_epsilon) {
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
                Op::Epsilon => {} // ε x = x ε = x
                Op::Concat(xs) => result.extend(xs.into_iter()),
                x => result.push(Expr(x)),
            }
        }
        match result.len() {
            0 => Expr::EPSILON,
            1 => result.into_iter().next().unwrap(),
            _ => Expr(Op::Concat(result)),
        }
    }
    /// Positive closure: `x+`.
    pub fn some(expr: Expr<A>) -> Expr<A> {
        Expr(match expr.0 {
            x @ Op::Epsilon => x, // ε+ = ε
            x @ Op::Some(_) => x, // (x+)+ = x+
            x => Op::Some(Box::new(Expr(x))),
        })
    }
    /// Optional: `x? = ε | x`.
    pub fn optional(expr: Expr<A>) -> Expr<A> {
        Expr::union([Expr::EPSILON, expr])
    }
    /// Closure, or the Kleene star: `x* = ε | x+`.
    pub fn many(expr: Expr<A>) -> Expr<A> {
        Expr::optional(Expr::some(expr))
    }
}

impl<A: Clone> Expr<A> {
    /// Calculate the positions of a regular expression as defined in Section 3.9.1 of *Dragon Book*.
    ///
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// let (te, n) = Expr::concat([
    ///     Expr::many(Expr::union([
    ///         Expr::singleton("a"),
    ///         Expr::singleton("b")
    ///     ])),
    ///     Expr::singleton("a"),
    ///     Expr::singleton("b"),
    ///     Expr::singleton("b"),
    ///     Expr::singleton("#"),
    /// ]).positions();
    /// assert_eq!(n, 6);
    /// assert_eq!(te, Expr::concat([
    ///     Expr::many(Expr::union([
    ///         Expr::singleton(("a", 0)),
    ///         Expr::singleton(("b", 1)),
    ///     ])),
    ///     Expr::singleton(("a", 2)),
    ///     Expr::singleton(("b", 3)),
    ///     Expr::singleton(("b", 4)),
    ///     Expr::singleton(("#", 5)),
    /// ]));
    /// ```
    pub fn positions(&self) -> (Expr<(A, usize)>, usize) {
        fn calc<A: Clone>(expr: &Expr<A>, n: &mut usize) -> Expr<(A, usize)> {
            Expr(match expr.deref() {
                Op::Epsilon => Op::Epsilon,
                Op::Singleton(a) => Op::Singleton((a.clone(), crate::utils::inc(n))),
                Op::Union(xs) => Op::Union(xs.iter().map(|x| calc(x, n)).collect()),
                Op::Concat(xs) => Op::Concat(xs.iter().map(|x| calc(x, n)).collect()),
                Op::Some(x) => Op::Some(Box::new(calc(x, n))),
            })
        }
        let mut count = 0;
        (calc(self, &mut count), count)
    }
}
