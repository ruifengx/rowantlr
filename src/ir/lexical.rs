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

//! IR for lexical specifications: use the [`Expr`] API to construct regular expressions.

pub mod char_class;

use std::collections::BTreeSet;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, Deref, DerefMut};
use crate::utils::IterHelper;

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
    /// Union: `L ∪ R` is regular if both `L` and `R` are regular.
    Union(Vec<R>),
    /// Concatenation: `L . R` is regular if both `L` and `R` are regular.
    Concat(Vec<R>),
    /// Positive closure: `L+` is regular if `L` is regular.
    Some(Box<R>),
}

impl<A> Expr<A> {
    /// Singleton: `a`.
    pub const fn singleton(a: A) -> Expr<A> { Expr(Op::Singleton(a)) }
    /// Empty string: `ε`.
    pub fn epsilon() -> Expr<A> { Expr::concat([]) }
    /// Union: `x₁ | x₂ | ... | xₙ`.
    ///
    /// If input has only one branch, no extra [`Op::Union`] layer is introduced:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// let e = Expr::from("abc");
    /// assert_eq!(e.clone(), Expr::union([e]));
    /// ```
    ///
    /// Nested [`Op::Union`] structures will be simplified:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// assert_eq!(Expr::union([
    ///     Expr::singleton('a'),
    ///     Expr::any_of("bc"),
    ///     Expr::singleton('d'),
    /// ]), Expr::any_of("abcd"));
    /// ```
    /// Note that [`Expr::any_of`] is just a convenient method for `Expr<char>`, a thin wrapper
    /// around [`Expr::union`] and [`Expr::singleton`]. Here is the fully expanded version:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// assert_eq!(Expr::union([
    ///     Expr::singleton('a'),
    ///     Expr::union([
    ///         Expr::singleton('b'),
    ///         Expr::singleton('c'),
    ///     ]),
    ///     Expr::singleton('d'),
    /// ]), Expr::union([
    ///     Expr::singleton('a'),
    ///     Expr::singleton('b'),
    ///     Expr::singleton('c'),
    ///     Expr::singleton('d'),
    /// ]));
    /// ```
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
        match result.len() {
            1 => result.into_iter().next().unwrap(),
            _ => Expr(Op::Union(result)),
        }
    }
    /// Concatenation: `x₁ x₂ ... xₙ`.
    ///
    /// If input has only one item, no extra [`Op::Concat`] layer is introduced:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// let e = Expr::singleton('a');
    /// assert_eq!(e.clone(), Expr::concat([e]));
    /// ```
    ///
    /// Nested [`Op::Concat`] structures will be simplified:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// assert_eq!(Expr::concat([
    ///     Expr::singleton('a'),
    ///     Expr::from("bc"),
    ///     Expr::singleton('d'),
    /// ]), Expr::from("abcd"));
    /// ```
    /// Note that [`Expr::from`] is just a convenient method for `Expr<char>`, a thin wrapper
    /// around [`Expr::concat`] and [`Expr::singleton`]. Here is the fully expanded version:
    /// ```
    /// # use rowantlr::ir::lexical::Expr;
    /// assert_eq!(Expr::concat([
    ///     Expr::singleton('a'),
    ///     Expr::concat([
    ///         Expr::singleton('b'),
    ///         Expr::singleton('c'),
    ///     ]),
    ///     Expr::singleton('d'),
    /// ]), Expr::concat([
    ///     Expr::singleton('a'),
    ///     Expr::singleton('b'),
    ///     Expr::singleton('c'),
    ///     Expr::singleton('d'),
    /// ]));
    /// ```
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
            Op::Union(xs) if xs.is_empty() => Op::Union(xs), // Ø+ = Ø
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

impl<A> BitOr for Expr<A> {
    type Output = Expr<A>;
    fn bitor(self, rhs: Self) -> Self::Output {
        Expr::union([self, rhs])
    }
}

impl<A> BitAnd for Expr<A> {
    type Output = Expr<A>;
    fn bitand(self, rhs: Self) -> Self::Output {
        Expr::concat([self, rhs])
    }
}

impl Expr<char> {
    /// Union of all the characters in some string.
    pub fn any_of(s: &str) -> Self {
        Expr::union_of(s.chars())
    }
}

impl<'a> From<&'a str> for Expr<char> {
    fn from(s: &str) -> Self {
        Expr::concat(s.chars().map(Expr::singleton))
    }
}

impl<A> Expr<A> {
    /// Union of all the elements in some iterator.
    pub fn union_of<I: IntoIterator<Item=A>>(s: I) -> Self {
        Expr::union(s.into_iter().map(Expr::singleton))
    }
}

impl<A, const N: usize> From<[A; N]> for Expr<A> {
    fn from(s: [A; N]) -> Self {
        Expr::concat(s.into_iter().map(Expr::singleton))
    }
}

/// Information of a (sub-) regular expression.
#[derive(Debug, Default, Clone, Eq, PartialEq)]
pub struct ExprInfo {
    /// Whether or not `e =>* ε`.
    pub nullable: bool,
    /// `firstpos(e)`: the set of positions in `e` which correspond to some symbol `a` such that
    /// `e =>* aβ`, i.e. the symbols appears in the first places of possible sentences in `L(e)`.
    pub first_pos: BTreeSet<u32>,
    /// `lastpos(e)`: the set of positions in `e` which correspond to some symbol `a` such that
    /// `e =>* βa`, i.e. the symbols appears in the last places of possible sentences in `L(e)`.
    pub last_pos: BTreeSet<u32>,
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
    pub fn singleton(pos: u32) -> ExprInfo {
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

impl BitOrAssign for ExprInfo {
    fn bitor_assign(&mut self, rhs: Self) {
        self.nullable = self.nullable || rhs.nullable;
        self.first_pos.extend(rhs.first_pos.iter().copied());
        self.last_pos.extend(rhs.last_pos.iter().copied());
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

impl BitAndAssign for ExprInfo {
    fn bitand_assign(&mut self, mut rhs: Self) {
        if self.nullable {
            self.first_pos.extend(rhs.first_pos.iter().copied());
        }
        std::mem::swap(&mut self.last_pos, &mut rhs.last_pos);
        if rhs.nullable {
            self.last_pos.extend(rhs.last_pos.iter().copied());
        }
        self.nullable = self.nullable && rhs.nullable;
    }
}

#[cfg(test)]
mod tests {
    use super::ExprInfo;

    fn input(nullable: bool, first_pos: impl IntoIterator<Item=u32>,
             last_pos: impl IntoIterator<Item=u32>) -> ExprInfo {
        ExprInfo {
            nullable,
            first_pos: first_pos.into_iter().collect(),
            last_pos: last_pos.into_iter().collect(),
        }
    }

    #[test]
    fn test_bit_op_assign() {
        let e1 = input(false, [0, 3], [42]);
        let e1null = input(true, [0, 3], [42]);
        let e2 = input(false, [6, 7], [100, 101]);
        let e2null = input(true, [6, 7], [100, 101]);
        fn prop(lhs: &ExprInfo, rhs: &ExprInfo) {
            assert_eq!(lhs.clone() & rhs.clone(), {
                let mut res = lhs.clone();
                res &= rhs.clone();
                res
            });
            assert_eq!(lhs.clone() | rhs.clone(), {
                let mut res = lhs.clone();
                res |= rhs.clone();
                res
            });
        }
        prop(&e1, &e2);
        prop(&e1null, &e2);
        prop(&e1, &e2null);
        prop(&e1null, &e2null);
    }
}

/// How is this position generated.
#[derive(Debug, Copy, Clone)]
pub enum PosInfo<A, Tag> {
    /// From a non-`ε` leaf node, i.e. an [`Op::Singleton`].
    Normal(A),
    /// From a phantom "accept" node, with a tag attached.
    Accept(Tag),
}

impl<A, Tag> PosInfo<A, Tag> {
    /// Converts from `&PosInfo<A, Tag>` to `PosInfo<&A, &Tag>`.
    pub fn as_ref(&self) -> PosInfo<&A, &Tag> {
        match self {
            PosInfo::Normal(a) => PosInfo::Normal(a),
            PosInfo::Accept(tag) => PosInfo::Accept(tag),
        }
    }
    /// Match against [`PosInfo::Normal`].
    pub fn into_normal(self) -> Option<A> {
        match self {
            PosInfo::Normal(a) => Some(a),
            PosInfo::Accept(_) => None,
        }
    }
    /// Match against [`PosInfo::Accept`].
    pub fn into_accept(self) -> Option<Tag> {
        match self {
            PosInfo::Normal(_) => None,
            PosInfo::Accept(tag) => Some(tag),
        }
    }
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
