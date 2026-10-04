"""
Gambit — A complete chess engine in Python.

Implements the full rules of classical chess:
  - All piece movement (pawn, knight, bishop, rook, queen, king)
  - Castling (kingside & queenside) with every legality condition
  - En passant captures
  - Pawn promotion (with piece choice)
  - Check, checkmate, and stalemate detection
  - Fifty-move rule, threefold repetition, insufficient material draws
  - Algebraic move notation (SAN) generation with disambiguation
  - Move history with undo/redo
  - FEN serialisation
  - A playable command-line interface

Usage:
    python chess.py              # start a game vs another human (CLI)
    python chess.py --help       # show commands

Author: CodeArts
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums and constants
# ---------------------------------------------------------------------------

class Color(Enum):
    WHITE = "w"
    BLACK = "b"

    @property
    def opponent(self) -> "Color":
        return Color.BLACK if self is Color.WHITE else Color.WHITE

    @property
    def pawn_dir(self) -> int:
        # White pawns move up the board (toward row 0); black toward row 7.
        return -1 if self is Color.WHITE else 1

    @property
    def back_rank(self) -> int:
        return 7 if self is Color.WHITE else 0

    @property
    def promo_rank(self) -> int:
        return 0 if self is Color.WHITE else 7

    def __str__(self) -> str:
        return "White" if self is Color.WHITE else "Black"


class PieceType(Enum):
    PAWN = "p"
    KNIGHT = "n"
    BISHOP = "b"
    ROOK = "r"
    QUEEN = "q"
    KING = "k"

    @property
    def value_int(self) -> int:
        return {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}[self.value]

    @property
    def glyph(self) -> str:
        return {"p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}[self.value]

    @classmethod
    def from_char(cls, ch: str) -> "PieceType":
        return cls(ch.lower())


@dataclass(frozen=True)
class Piece:
    type: PieceType
    color: Color

    @property
    def is_white(self) -> bool:
        return self.color is Color.WHITE

    @property
    def fen_char(self) -> str:
        return self.type.value.upper() if self.is_white else self.type.value

    @classmethod
    def from_fen(cls, ch: str) -> "Piece":
        return cls(PieceType.from_char(ch), Color.WHITE if ch.isupper() else Color.BLACK)

    def __str__(self) -> str:
        g = self.type.glyph
        return g if self.color is Color.BLACK else g.upper()


# ---------------------------------------------------------------------------
# Squares and moves
# ---------------------------------------------------------------------------

FILES = "abcdefgh"

def square_name(row: int, col: int) -> str:
    """Convert (row, col) board coordinates to algebraic name, e.g. (6,0)->'a2'."""
    return FILES[col] + str(8 - row)

def parse_square(name: str) -> tuple[int, int]:
    """Parse 'e4' -> (row, col). Raises ValueError on bad input."""
    if len(name) != 2 or name[0] not in FILES or name[1] not in "12345678":
        raise ValueError(f"Invalid square: {name!r}")
    return (8 - int(name[1]), FILES.index(name[0]))


@dataclass
class Move:
    from_sq: tuple[int, int]
    to_sq: tuple[int, int]
    piece: Piece
    captured: Optional[Piece] = None
    is_castle: Optional[str] = None      # 'K' or 'Q' or None
    is_en_passant: bool = False
    is_double_pawn: bool = False
    promotion: Optional[PieceType] = None

    @property
    def uci(self) -> str:
        s = square_name(*self.from_sq) + square_name(*self.to_sq)
        if self.promotion:
            s += self.promotion.value
        return s

    def __str__(self) -> str:
        return self.uci


# ---------------------------------------------------------------------------
# Castling rights
# ---------------------------------------------------------------------------

@dataclass
class CastlingRights:
    wK: bool = True
    wQ: bool = True
    bK: bool = True
    bQ: bool = True

    def copy(self) -> "CastlingRights":
        return CastlingRights(self.wK, self.wQ, self.bK, self.bQ)

    def fen(self) -> str:
        s = ""
        if self.wK: s += "K"
        if self.wQ: s += "Q"
        if self.bK: s += "k"
        if self.bQ: s += "q"
        return s or "-"


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Knight and king offsets
KNIGHT_OFFSETS = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
KING_OFFSETS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
ORTH_DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
DIAG_DIRS = [(-1,-1),(-1,1),(1,-1),(1,1)]


class Board:
    """An 8x8 board with full move generation and rule enforcement."""

    def __init__(self, fen: str = INITIAL_FEN):
        self.squares: list[list[Optional[Piece]]] = [[None]*8 for _ in range(8)]
        self.turn: Color = Color.WHITE
        self.castling = CastlingRights()
        self.en_passant: Optional[tuple[int,int]] = None
        self.halfmove = 0
        self.fullmove = 1
        self.load_fen(fen)

    # ---- FEN --------------------------------------------------------------
    def load_fen(self, fen: str) -> None:
        parts = fen.split()
        rows = parts[0].split("/")
        self.squares = [[None] * 8 for _ in range(8)]
        for r, row in enumerate(rows):
            c = 0
            for ch in row:
                if ch.isdigit():
                    c += int(ch)
                else:
                    self.squares[r][c] = Piece.from_fen(ch)
                    c += 1
        self.turn = Color.WHITE if parts[1] == "w" else Color.BLACK
        cr = parts[2]
        self.castling = CastlingRights("K" in cr, "Q" in cr, "k" in cr, "q" in cr)
        self.en_passant = parse_square(parts[3]) if parts[3] != "-" else None
        self.halfmove = int(parts[4]) if len(parts) > 4 else 0
        self.fullmove = int(parts[5]) if len(parts) > 5 else 1

    def fen(self) -> str:
        rows = []
        for r in range(8):
            row, empty = "", 0
            for c in range(8):
                p = self.squares[r][c]
                if p is None:
                    empty += 1
                else:
                    if empty:
                        row += str(empty); empty = 0
                    row += p.fen_char
            if empty: row += str(empty)
            rows.append(row)
        ep = square_name(*self.en_passant) if self.en_passant else "-"
        return f"{'/'.join(rows)} {self.turn.value} {self.castling.fen()} {ep} {self.halfmove} {self.fullmove}"

    # ---- Helpers ----------------------------------------------------------
    def get(self, r: int, c: int) -> Optional[Piece]:
        return self.squares[r][c]

    def set(self, r: int, c: int, p: Optional[Piece]) -> None:
        self.squares[r][c] = p

    @staticmethod
    def _in(r: int, c: int) -> bool:
        return 0 <= r < 8 and 0 <= c < 8

    def find_king(self, color: Color) -> tuple[int,int]:
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p and p.type is PieceType.KING and p.color is color:
                    return (r, c)
        raise ValueError(f"No {color} king on board")

    # ---- Attack detection ------------------------------------------------
    def is_attacked(self, r: int, c: int, by: Color) -> bool:
        """Is square (r,c) attacked by any piece of color `by`?"""
        sq = self.squares
        # Pawns
        pr = r - by.pawn_dir
        for dc in (-1, 1):
            pc = c + dc
            if self._in(pr, pc):
                p = sq[pr][pc]
                if p and p.color is by and p.type is PieceType.PAWN:
                    return True
        # Knights
        for dr, dc in KNIGHT_OFFSETS:
            nr, nc = r+dr, c+dc
            if self._in(nr, nc):
                p = sq[nr][nc]
                if p and p.color is by and p.type is PieceType.KNIGHT:
                    return True
        # King
        for dr, dc in KING_OFFSETS:
            nr, nc = r+dr, c+dc
            if self._in(nr, nc):
                p = sq[nr][nc]
                if p and p.color is by and p.type is PieceType.KING:
                    return True
        # Orthogonal sliders (rook, queen)
        for dr, dc in ORTH_DIRS:
            nr, nc = r+dr, c+dc
            while self._in(nr, nc):
                p = sq[nr][nc]
                if p:
                    if p.color is by and p.type in (PieceType.ROOK, PieceType.QUEEN):
                        return True
                    break
                nr += dr; nc += dc
        # Diagonal sliders (bishop, queen)
        for dr, dc in DIAG_DIRS:
            nr, nc = r+dr, c+dc
            while self._in(nr, nc):
                p = sq[nr][nc]
                if p:
                    if p.color is by and p.type in (PieceType.BISHOP, PieceType.QUEEN):
                        return True
                    break
                nr += dr; nc += dc
        return False

    def in_check(self, color: Color) -> bool:
        kr, kc = self.find_king(color)
        return self.is_attacked(kr, kc, color.opponent)

    # ---- Pseudo-legal move generation ------------------------------------
    def _pseudo_from(self, r: int, c: int) -> list[Move]:
        p = self.squares[r][c]
        if p is None:
            return []
        color, pt = p.color, p.type
        moves: list[Move] = []
        sq = self.squares

        def add(nr, nc, **kw):
            cap = sq[nr][nc] if not kw.get("is_en_passant") else None
            moves.append(Move((r,c),(nr,nc),p,captured=cap,**kw))

        if pt is PieceType.PAWN:
            dir = color.pawn_dir
            start = color.back_rank - dir   # row where pawns start... actually:
            # White pawns start row 6, black row 1.
            start_row = 6 if color is Color.WHITE else 1
            promo_row = color.promo_rank
            # Forward one
            nr = r + dir
            if self._in(nr, c) and sq[nr][c] is None:
                if nr == promo_row:
                    for promo in (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT):
                        moves.append(Move((r,c),(nr,c),p,promotion=promo))
                else:
                    add(nr, c)
                    # Forward two
                    nr2 = r + 2*dir
                    if r == start_row and sq[nr2][c] is None:
                        add(nr2, c, is_double_pawn=True)
            # Captures
            for dc in (-1, 1):
                nc = c + dc
                if not self._in(nr, nc):
                    continue
                tgt = sq[nr][nc]
                if tgt and tgt.color is not color:
                    if nr == promo_row:
                        for promo in (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT):
                            moves.append(Move((r,c),(nr,nc),p,captured=tgt,promotion=promo))
                    else:
                        add(nr, nc)
                # En passant
                if self.en_passant and self.en_passant == (nr, nc):
                    ep_cap = sq[r][nc]
                    moves.append(Move((r,c),(nr,nc),p,captured=ep_cap,is_en_passant=True))

        elif pt is PieceType.KNIGHT:
            for dr, dc in KNIGHT_OFFSETS:
                nr, nc = r+dr, c+dc
                if self._in(nr, nc) and (sq[nr][nc] is None or sq[nr][nc].color is not color):
                    add(nr, nc)

        elif pt is PieceType.KING:
            for dr, dc in KING_OFFSETS:
                nr, nc = r+dr, c+dc
                if self._in(nr, nc) and (sq[nr][nc] is None or sq[nr][nc].color is not color):
                    add(nr, nc)
            # Castling
            home = color.back_rank
            if r == home and c == 4:
                opp = color.opponent
                can_k = self.castling.wK if color is Color.WHITE else self.castling.bK
                can_q = self.castling.wQ if color is Color.WHITE else self.castling.bQ
                # Kingside: squares 5,6 empty; rook on 7; king not passing through check
                if (can_k and sq[home][5] is None and sq[home][6] is None
                        and sq[home][7] and sq[home][7].type is PieceType.ROOK
                        and not self.is_attacked(home, 4, opp)
                        and not self.is_attacked(home, 5, opp)
                        and not self.is_attacked(home, 6, opp)):
                    add(home, 6, is_castle="K")
                # Queenside: squares 1,2,3 empty; rook on 0
                if (can_q and sq[home][1] is None and sq[home][2] is None and sq[home][3] is None
                        and sq[home][0] and sq[home][0].type is PieceType.ROOK
                        and not self.is_attacked(home, 4, opp)
                        and not self.is_attacked(home, 3, opp)
                        and not self.is_attacked(home, 2, opp)):
                    add(home, 2, is_castle="Q")

        else:
            dirs = []
            if pt in (PieceType.ROOK, PieceType.QUEEN): dirs += ORTH_DIRS
            if pt in (PieceType.BISHOP, PieceType.QUEEN): dirs += DIAG_DIRS
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                while self._in(nr, nc):
                    tgt = sq[nr][nc]
                    if tgt is None:
                        add(nr, nc)
                    else:
                        if tgt.color is not color:
                            add(nr, nc)
                        break
                    nr += dr; nc += dc
        return moves

    def _apply(self, mv: Move) -> None:
        """Apply a move in-place (does not validate)."""
        r1, c1 = mv.from_sq
        r2, c2 = mv.to_sq
        p = mv.piece
        self.squares[r1][c1] = None
        if mv.is_en_passant:
            self.squares[r1][c2] = None
        if mv.promotion:
            self.squares[r2][c2] = Piece(mv.promotion, p.color)
        else:
            self.squares[r2][c2] = p
        if mv.is_castle == "K":
            self.squares[r2][5] = self.squares[r2][7]
            self.squares[r2][7] = None
        elif mv.is_castle == "Q":
            self.squares[r2][3] = self.squares[r2][0]
            self.squares[r2][0] = None

    def _undo_apply(self, mv: Move, captured: Optional[Piece],
                    ep: Optional[tuple[int,int]], castling: CastlingRights,
                    halfmove: int, fullmove: int, turn: Color) -> None:
        r1, c1 = mv.from_sq
        r2, c2 = mv.to_sq
        p = mv.piece
        self.squares[r1][c1] = p
        if mv.promotion:
            self.squares[r2][c2] = None
        if mv.is_en_passant:
            self.squares[r2][c2] = None
            self.squares[r1][c2] = captured
        else:
            self.squares[r2][c2] = captured
        if mv.is_castle == "K":
            self.squares[r2][7] = self.squares[r2][5]
            self.squares[r2][5] = None
        elif mv.is_castle == "Q":
            self.squares[r2][0] = self.squares[r2][3]
            self.squares[r2][3] = None
        self.en_passant = ep
        self.castling = castling
        self.halfmove = halfmove
        self.fullmove = fullmove
        self.turn = turn

    # ---- Legal moves -----------------------------------------------------
    def legal_moves_from(self, r: int, c: int) -> list[Move]:
        p = self.squares[r][c]
        if p is None or p.color is not self.turn:
            return []
        result = []
        for mv in self._pseudo_from(r, c):
            self._apply(mv)
            if not self.in_check(p.color):
                result.append(mv)
            self._unapply(mv)
        return result

    def _unapply(self, mv: Move) -> None:
        """Reverse a move that was applied by _apply (no state saved)."""
        r1, c1 = mv.from_sq
        r2, c2 = mv.to_sq
        p = mv.piece
        # Restore moved piece
        self.squares[r1][c1] = p
        if mv.promotion:
            self.squares[r2][c2] = None
        # Restore capture
        if mv.is_en_passant:
            self.squares[r2][c2] = None
            self.squares[r1][c2] = mv.captured
        else:
            self.squares[r2][c2] = mv.captured
        # Undo castling rook
        if mv.is_castle == "K":
            self.squares[r2][7] = self.squares[r2][5]
            self.squares[r2][5] = None
        elif mv.is_castle == "Q":
            self.squares[r2][0] = self.squares[r2][3]
            self.squares[r2][3] = None

    def all_legal_moves(self, color: Optional[Color] = None) -> list[Move]:
        color = color or self.turn
        moves = []
        for r in range(8):
            for c in range(8):
                p = self.squares[r][c]
                if p and p.color is color:
                    for mv in self._pseudo_from(r, c):
                        self._apply(mv)
                        if not self.in_check(color):
                            moves.append(mv)
                        self._unapply(mv)
        return moves

    # ---- Game status -----------------------------------------------------
    def is_checkmate(self) -> bool:
        return self.in_check(self.turn) and not self.all_legal_moves()

    def is_stalemate(self) -> bool:
        return not self.in_check(self.turn) and not self.all_legal_moves()

    def is_insufficient_material(self) -> bool:
        pieces = [p for row in self.squares for p in row if p]
        non_kings = [p for p in pieces if p.type is not PieceType.KING]
        # K vs K
        if len(non_kings) == 0:
            return True
        # K + minor vs K
        if len(non_kings) == 1 and non_kings[0].type in (PieceType.KNIGHT, PieceType.BISHOP):
            return True
        # K + B vs K + B, same color bishops
        if len(non_kings) == 2 and all(p.type is PieceType.BISHOP for p in non_kings):
            colors = set()
            for r in range(8):
                for c in range(8):
                    p = self.squares[r][c]
                    if p and p.type is PieceType.BISHOP:
                        colors.add((r + c) % 2)
            if len(colors) == 1:
                return True
        return False


# ---------------------------------------------------------------------------
# Game — history, SAN notation, undo, draw detection
# ---------------------------------------------------------------------------

@dataclass
class HistoryEntry:
    move: Move
    san: str
    captured: Optional[Piece]
    castling_before: CastlingRights
    en_passant_before: Optional[tuple[int,int]]
    halfmove_before: int
    fullmove_before: int
    fen_before: str


class Game:
    def __init__(self, fen: str = INITIAL_FEN):
        self.board = Board(fen)
        self.history: list[HistoryEntry] = None  # type: ignore
        self.history = []
        self.position_counts: dict[str, int] = {}
        self._count_position()
        self.game_over: bool = False
        self.result: Optional[str] = None  # e.g. "1-0", "0-1", "1/2-1/2"

    def _count_position(self) -> None:
        key = self.board.fen().split(" ")[0] + self.board.fen().split(" ")[1]
        self.position_counts[key] = self.position_counts.get(key, 0) + 1

    def _uncount_position(self, key: str) -> None:
        self.position_counts[key] -= 1
        if self.position_counts[key] <= 0:
            del self.position_counts[key]

    # ---- Move execution --------------------------------------------------
    def make_move(self, mv: Move) -> str:
        """Apply a legal move, update history, return SAN string."""
        if self.game_over:
            raise RuntimeError("Game is over")

        fen_before = self.board.fen()
        castling_before = self.board.castling.copy()
        ep_before = self.board.en_passant
        hm_before = self.board.halfmove
        fm_before = self.board.fullmove
        captured = mv.captured

        # Build SAN before applying (needs board state for disambiguation)
        san = self._san(mv)

        # Update castling rights
        cr = self.board.castling
        p = mv.piece
        if p.type is PieceType.KING:
            if p.color is Color.WHITE: cr.wK = cr.wQ = False
            else: cr.bK = cr.bQ = False
        if p.type is PieceType.ROOK:
            if p.color is Color.WHITE and mv.from_sq == (7,0): cr.wQ = False
            if p.color is Color.WHITE and mv.from_sq == (7,7): cr.wK = False
            if p.color is Color.BLACK and mv.from_sq == (0,0): cr.bQ = False
            if p.color is Color.BLACK and mv.from_sq == (0,7): cr.bK = False
        # Rook captured on home square
        if mv.to_sq == (7,0): cr.wQ = False
        if mv.to_sq == (7,7): cr.wK = False
        if mv.to_sq == (0,0): cr.bQ = False
        if mv.to_sq == (0,7): cr.bK = False

        # En passant target
        self.board.en_passant = None
        if mv.is_double_pawn:
            r1, c1 = mv.from_sq; r2, c2 = mv.to_sq
            self.board.en_passant = ((r1+r2)//2, c1)

        # Halfmove clock
        if p.type is PieceType.PAWN or captured:
            self.board.halfmove = 0
        else:
            self.board.halfmove += 1

        # Apply
        self.board._apply(mv)

        # Fullmove
        if p.color is Color.BLACK:
            self.board.fullmove += 1
        self.board.turn = p.color.opponent

        # Check/mate suffix
        opp = self.board.turn
        opp_in_check = self.board.in_check(opp)
        if opp_in_check and not self.board.all_legal_moves():
            san += "#"
        elif opp_in_check:
            san += "+"

        self.history.append(HistoryEntry(
            move=mv, san=san, captured=captured,
            castling_before=castling_before, en_passant_before=ep_before,
            halfmove_before=hm_before, fullmove_before=fm_before,
            fen_before=fen_before,
        ))
        self._count_position()
        self._check_terminal()
        return san

    def undo(self) -> Optional[Move]:
        if not self.history:
            return None
        entry = self.history.pop()
        # Uncount current position
        key = self.board.fen().split(" ")[0] + self.board.fen().split(" ")[1]
        self._uncount_position(key)
        # Restore board state
        self.board.load_fen(entry.fen_before)
        self.game_over = False
        self.result = None
        return entry.move

    # ---- SAN notation ----------------------------------------------------
    def _san(self, mv: Move) -> str:
        if mv.is_castle == "K":
            return "O-O"
        if mv.is_castle == "Q":
            return "O-O-O"
        p = mv.piece
        r1, c1 = mv.from_sq
        r2, c2 = mv.to_sq
        s = ""
        if p.type is not PieceType.PAWN:
            s += p.type.value.upper()
            # Disambiguation
            ambiguous = []
            for r in range(8):
                for c in range(8):
                    if (r,c) == (r1,c1): continue
                    q = self.board.squares[r][c]
                    if q and q == p:
                        for m in self.board.legal_moves_from(r, c):
                            if m.to_sq == (r2, c2):
                                ambiguous.append((r, c))
            if ambiguous:
                same_file = any(c == c1 for r, c in ambiguous)
                same_rank = any(r == r1 for r, c in ambiguous)
                if not same_file:
                    s += FILES[c1]
                elif not same_rank:
                    s += str(8 - r1)
                else:
                    s += square_name(r1, c1)
        if p.type is PieceType.PAWN and (mv.captured or mv.is_en_passant):
            s += FILES[c1]
        if mv.captured or mv.is_en_passant:
            s += "x"
        s += square_name(r2, c2)
        if mv.promotion:
            s += "=" + mv.promotion.value.upper()
        return s

    # ---- Move parsing ----------------------------------------------------
    def parse_san(self, text: str) -> Move:
        """Parse a SAN or UCI move string into a legal Move."""
        text = text.strip().rstrip("+#!?")
        # Castling
        if text in ("O-O", "0-0", "OO"):
            return self._find_castle("K")
        if text in ("O-O-O", "0-0-0", "OOO"):
            return self._find_castle("Q")
        # Promotion suffix
        promo = None
        if "=" in text:
            base, promo_ch = text.split("=")
            promo = PieceType.from_char(promo_ch[0])
            text = base
        elif len(text) == 5 and text[-1] in "qrbn":
            promo = PieceType.from_char(text[-1])
            text = text[:-1]
        # UCI like e2e4
        if len(text) == 4 and text[0] in FILES and text[1] in "12345678":
            return self._find_uci(text, promo)
        # SAN: strip piece letter
        piece_type = PieceType.PAWN
        if text[0] in "NBRQK":
            piece_type = PieceType.from_char(text[0])
            text = text[1:]
        # Capture marker
        capture = "x" in text
        text = text.replace("x", "")
        # Destination is last two chars
        dest = text[-2:]
        r2, c2 = parse_square(dest)
        # Disambiguation in between
        disamb = text[:-2]
        dis_file = disamb[0] if disamb and disamb[0] in FILES else None
        dis_rank = int(disamb[0]) if disamb and disamb[0] in "12345678" else None
        if len(disamb) == 2:
            dis_file = disamb[0] if disamb[0] in FILES else None
            dis_rank = int(disamb[1]) if disamb[1] in "12345678" else None
        # Find matching legal move
        candidates = []
        for r in range(8):
            for c in range(8):
                p = self.board.squares[r][c]
                if not p or p.color is not self.board.turn or p.type is not piece_type:
                    continue
                if dis_file is not None and FILES[c] != dis_file:
                    continue
                if dis_rank is not None and (8 - r) != dis_rank:
                    continue
                for m in self.board.legal_moves_from(r, c):
                    if m.to_sq == (r2, c2):
                        if promo and m.promotion is not promo:
                            continue
                        if not promo and m.promotion:
                            continue
                        candidates.append(m)
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise ValueError(f"Illegal or unrecognised move: {text!r}")
        # Prefer queen promotion if ambiguous
        for m in candidates:
            if m.promotion is None or m.promotion is PieceType.QUEEN:
                return m
        return candidates[0]

    def _find_castle(self, side: str) -> Move:
        home = self.board.turn.back_rank
        for m in self.board.legal_moves_from(home, 4):
            if m.is_castle == side:
                return m
        raise ValueError(f"Illegal castling: {side}")

    def _find_uci(self, text: str, promo: Optional[PieceType]) -> Move:
        r1, c1 = parse_square(text[:2])
        r2, c2 = parse_square(text[2:4])
        for m in self.board.legal_moves_from(r1, c1):
            if m.to_sq == (r2, c2):
                if promo and m.promotion is not promo:
                    continue
                return m
        raise ValueError(f"Illegal UCI move: {text!r}")

    # ---- Terminal detection ----------------------------------------------
    def _check_terminal(self) -> None:
        if self.board.is_checkmate():
            self.game_over = True
            self.result = "1-0" if self.board.turn is Color.BLACK else "0-1"
        elif self.board.is_stalemate():
            self.game_over = True
            self.result = "1/2-1/2"
        elif self.board.halfmove >= 100:
            self.game_over = True
            self.result = "1/2-1/2"
        elif self.board.is_insufficient_material():
            self.game_over = True
            self.result = "1/2-1/2"
        else:
            key = self.board.fen().split(" ")[0] + self.board.fen().split(" ")[1]
            if self.position_counts.get(key, 0) >= 3:
                self.game_over = True
                self.result = "1/2-1/2"

    def status(self) -> str:
        if self.game_over:
            if self.result == "1-0": return "Checkmate — White wins"
            if self.result == "0-1": return "Checkmate — Black wins"
            return "Draw"
        if self.board.in_check(self.board.turn):
            return f"{self.board.turn} is in check"
        return f"{self.board.turn} to move"

    # ---- Pretty print ----------------------------------------------------
    def display(self) -> str:
        lines = []
        lines.append("    a   b   c   d   e   f   g   h")
        lines.append("  ┌───┬───┬───┬───┬───┬───┬───┬───┐")
        for r in range(8):
            row_chars = []
            for c in range(8):
                p = self.board.squares[r][c]
                if p is None:
                    row_chars.append("   ")
                else:
                    g = p.type.glyph
                    row_chars.append(f" {g} ")
            lines.append(f"{8-r} │" + "│".join(row_chars) + f"│ {8-r}")
            if r < 7:
                lines.append("  ├───┼───┼───┼───┼───┼───┼───┼───┤")
        lines.append("  └───┴───┴───┴───┴───┴───┴───┴───┘")
        lines.append("    a   b   c   d   e   f   g   h")
        lines.append("")
        lines.append(f"  Turn: {self.board.turn}   Castling: {self.board.castling.fen()}   "
                     f"EP: {square_name(*self.board.en_passant) if self.board.en_passant else '-'}   "
                     f"FEN: {self.board.fen()}")
        if self.history:
            moves_str = "  ".join(f"{i//2+1}.{' ' if i%2==0 else '..'} {e.san}"
                                   for i, e in enumerate(self.history))
            lines.append(f"  Moves: {moves_str}")
        lines.append(f"  {self.status()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

HELP_TEXT = """
Commands:
  <move>       Play a move, e.g. e4, Nf3, exd5, O-O, e8=Q, or UCI e2e4
  undo         Take back the last move
  moves        List all legal moves
  fen          Show the current FEN
  display      Redraw the board
  reset        Start a new game
  help         Show this help
  quit / exit  Leave the game
""".strip()


def play_cli() -> None:
    game = Game()
    print("\n  ♚ Gambit — Chess ♔\n")
    print(game.display())
    print("\n  Enter moves in algebraic notation (e.g. e4, Nf3, O-O). Type 'help' for commands.\n")

    while True:
        try:
            prompt = f"  {game.board.turn}> "
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.\n")
            return

        if not raw:
            continue
        cmd = raw.lower()

        if cmd in ("quit", "exit"):
            print("  Goodbye.\n")
            return
        if cmd == "help":
            print(HELP_TEXT); continue
        if cmd == "reset":
            game = Game()
            print("\n  New game started.\n"); print(game.display()); continue
        if cmd == "undo":
            mv = game.undo()
            if mv: print(f"  Undone: {mv}\n")
            else: print("  Nothing to undo.\n")
            print(game.display()); continue
        if cmd == "moves":
            ms = game.board.all_legal_moves()
            sans = [game._san(m) for m in ms]
            print(f"  Legal moves ({len(ms)}): {', '.join(sans)}\n")
            continue
        if cmd == "fen":
            print(f"  {game.board.fen()}\n"); continue
        if cmd == "display":
            print(game.display()); continue

        try:
            mv = game.parse_san(raw)
            san = game.make_move(mv)
            print(f"\n  {mv.piece.color} played {san}\n")
            print(game.display())
            if game.game_over:
                print(f"\n  ★ Game over: {game.status()} ★\n")
                return
        except ValueError as e:
            print(f"  ✗ {e}\n")


# ---------------------------------------------------------------------------
# Chess AI — minimax with alpha-beta pruning
# ---------------------------------------------------------------------------

import random

# Material values in centipawns
AI_MATERIAL = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 20000}

# Piece-square tables (from white's perspective, row 0 = rank 8)
PST_PAWN = [
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [ 50, 50, 50, 50, 50, 50, 50, 50],
    [ 10, 10, 20, 30, 30, 20, 10, 10],
    [  5,  5, 10, 25, 25, 10,  5,  5],
    [  0,  0,  0, 20, 20,  0,  0,  0],
    [  5, -5,-10,  0,  0,-10, -5,  5],
    [  5, 10, 10,-20,-20, 10, 10,  5],
    [  0,  0,  0,  0,  0,  0,  0,  0],
]
PST_KNIGHT = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50],
]
PST_BISHOP = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5,  5,  5,  5,  5,-10],
    [-10,  0,  5,  0,  0,  5,  0,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20],
]
PST_ROOK = [
    [  0,  0,  0,  0,  0,  0,  0,  0],
    [  5, 10, 10, 10, 10, 10, 10,  5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [ -5,  0,  0,  0,  0,  0,  0, -5],
    [  0,  0,  0,  5,  5,  0,  0,  0],
]
PST_QUEEN = [
    [-20,-10,-10, -5, -5,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5,  5,  5,  5,  0,-10],
    [ -5,  0,  5,  5,  5,  5,  0, -5],
    [  0,  0,  5,  5,  5,  5,  0, -5],
    [-10,  5,  5,  5,  5,  5,  0,-10],
    [-10,  0,  5,  0,  0,  0,  0,-10],
    [-20,-10,-10, -5, -5,-10,-10,-20],
]
PST_KING = [
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-30,-40,-40,-50,-50,-40,-40,-30],
    [-20,-30,-30,-40,-40,-30,-30,-20],
    [-10,-20,-20,-20,-20,-20,-20,-10],
    [ 20, 20,  0,  0,  0,  0, 20, 20],
    [ 20, 30, 10,  0,  0, 10, 30, 20],
]
PST = {"p": PST_PAWN, "n": PST_KNIGHT, "b": PST_BISHOP, "r": PST_ROOK, "q": PST_QUEEN, "k": PST_KING}

MATE_SCORE = 1000000


class ChessAI:
    """A simple chess AI using minimax with alpha-beta pruning."""

    def __init__(self, depth: int = 3):
        self.depth = depth

    def evaluate(self, board: Board) -> int:
        """Static evaluation from white's perspective."""
        score = 0
        for r in range(8):
            for c in range(8):
                p = board.squares[r][c]
                if p is None:
                    continue
                val = AI_MATERIAL[p.type.value]
                pst = PST[p.type.value]
                if p.color is Color.WHITE:
                    score += val + pst[r][c]
                else:
                    score -= val + pst[7 - r][c]
        return score

    def best_move(self, game: Game) -> Optional[Move]:
        """Find the best move for the side to move."""
        board = game.board
        color = board.turn
        moves = board.all_legal_moves(color)
        if not moves:
            return None

        # Shuffle for variety in equal positions
        random.shuffle(moves)

        best_mv = moves[0]
        alpha, beta = -MATE_SCORE * 2, MATE_SCORE * 2
        sign = 1 if color is Color.WHITE else -1
        best_score = -MATE_SCORE * 2

        for mv in moves:
            board._apply(mv)
            score = -sign * self._minimax(board, self.depth - 1, -beta, -alpha, color.opponent)
            board._unapply(mv)
            if score > best_score:
                best_score = score
                best_mv = mv
            if score > alpha:
                alpha = score

        return best_mv

    def _minimax(self, board: Board, depth: int, alpha: int, beta: int, color: Color) -> int:
        """Negamax with alpha-beta pruning. Returns score from `color`'s perspective."""
        if depth == 0:
            eval_w = self.evaluate(board)
            return eval_w if color is Color.WHITE else -eval_w

        moves = board.all_legal_moves(color)
        if not moves:
            if board.in_check(color):
                return -MATE_SCORE - depth  # checkmate (prefer later mates)
            return 0  # stalemate

        best = -MATE_SCORE * 2
        for mv in moves:
            board._apply(mv)
            score = -self._minimax(board, depth - 1, -beta, -alpha, color.opponent)
            board._unapply(mv)
            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best


# ---------------------------------------------------------------------------
# Native GUI — polished Tkinter chess interface (no external dependencies)
# ---------------------------------------------------------------------------

import tkinter as tk
from tkinter import font as tkfont
import threading

# Colour palette — warm "chess club" aesthetic
C_BG       = "#1a1410"
C_BG2      = "#221a14"
C_PANEL    = "#2a2018"
C_PANEL2   = "#332720"
C_LINE     = "#3d2f22"
C_INK      = "#f3ead8"
C_INK_DIM  = "#b8a888"
C_INK_FAINT= "#7d6f56"
C_GOLD     = "#d4af78"
C_GOLD_B   = "#e8c890"
C_AMBER    = "#e0a854"
C_CRIMSON  = "#c85a4a"
C_TEAL     = "#5fb8a6"
C_LSQ      = "#e8d4ad"
C_DSQ      = "#8a5a3a"
C_LSQ_HI   = "#f0e0bd"
C_DSQ_HI   = "#a06a48"
C_SEL      = "#e8c890"
C_LAST     = "#c89b5e"
C_CHECK    = "#c85a4a"

PIECE_GLYPH = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265a", "q": "\u265b", "r": "\u265c", "b": "\u265d", "n": "\u265e", "p": "\u265f",
}
PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
FILES_LIST = "abcdefgh"


class ChessGUI:
    """A polished Tkinter front-end for the chess engine."""

    SQ = 64          # square pixel size
    BOARD = SQ * 8   # board pixel size (512)

    def __init__(self):
        self.game = Game()
        self.selected = None
        self.legal_targets = []
        self.flipped = False
        self.pending_promo = None

        # AI / mode state
        self.ai_opponent = False       # True if playing vs computer
        self.ai_color = Color.BLACK    # which color the AI plays
        self.ai_depth = 3             # search depth
        self.ai = None                # ChessAI instance
        self.ai_thinking = False      # AI is computing
        self.ai_thread = None

        self.root = tk.Tk()
        self.root.title("Gambit — Chess")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        # Try to find a font that renders chess glyphs well
        piece_family = "Segoe UI Symbol"
        for fam in ("Segoe UI Symbol", "DejaVu Sans", "Arial Unicode MS", "Noto Sans"):
            if fam in tkfont.families():
                piece_family = fam
                break
        self.piece_font = tkfont.Font(family=piece_family, size=36)
        self.coord_font = tkfont.Font(family="Inter", size=9, weight="bold")
        self.title_font = tkfont.Font(family="Helvetica", size=18, weight="bold")
        self.label_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self.text_font = tkfont.Font(family="Helvetica", size=11)
        self.small_font = tkfont.Font(family="Helvetica", size=9)
        self.mono_font = tkfont.Font(family="Consolas", size=11)
        self.logo_font = tkfont.Font(family=piece_family, size=28)

        self._build_layout()
        self.root.bind("<Key>", self._on_key)

        # Show mode selection on first start
        self.root.after(100, self._show_start_screen)

    # ---- Layout -----------------------------------------------------------
    def _build_layout(self):
        # Root container
        container = tk.Frame(self.root, bg=C_BG, padx=24, pady=24)
        container.pack()

        # Left: board area
        left = tk.Frame(container, bg=C_BG)
        left.pack(side=tk.LEFT, padx=(0, 20))

        # Header with chess logo
        header = tk.Frame(left, bg=C_BG)
        header.pack(fill=tk.X, pady=(0, 12))

        # Chess logo — golden badge with king crown
        logo_canvas = tk.Canvas(header, width=48, height=48, bg=C_BG, highlightthickness=0)
        logo_canvas.pack(side=tk.LEFT)
        logo_canvas.create_oval(2, 2, 46, 46, fill=C_GOLD, outline=C_AMBER, width=2)
        logo_canvas.create_oval(6, 6, 42, 42, fill="", outline=C_GOLD_B, width=1)
        logo_canvas.create_text(24, 25, text="\u265a", fill="#2a1c10", font=self.logo_font)

        title_box = tk.Frame(header, bg=C_BG)
        title_box.pack(side=tk.LEFT, padx=(10, 0))
        tk.Label(title_box, text="Gambit", fg=C_INK, bg=C_BG,
                 font=self.title_font).pack(anchor="w")
        tk.Label(title_box, text="CLASSICAL CHESS", fg=C_INK_FAINT, bg=C_BG,
                 font=self.small_font).pack(anchor="w")

        # Turn pill
        self.turn_frame = tk.Frame(header, bg=C_PANEL, highlightbackground=C_LINE,
                                   highlightthickness=1)
        self.turn_frame.pack(side=tk.RIGHT)
        self.turn_dot = tk.Canvas(self.turn_frame, width=16, height=16, bg=C_PANEL,
                                  highlightthickness=0)
        self.turn_dot.pack(side=tk.LEFT, padx=(10, 6), pady=6)
        self.turn_label = tk.Label(self.turn_frame, text="White to move", fg=C_INK,
                                   bg=C_PANEL, font=self.label_font)
        self.turn_label.pack(side=tk.LEFT, padx=(0, 12))

        # Board canvas with decorative frame
        frame_wrap = tk.Frame(left, bg="#241a12", highlightbackground=C_LINE,
                              highlightthickness=2)
        frame_wrap.pack()
        self.canvas = tk.Canvas(frame_wrap, width=self.BOARD, height=self.BOARD,
                                bg="#241a12", highlightthickness=0)
        self.canvas.pack(padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_click)

        # Bottom row: status (left) + credit (right)
        bottom = tk.Frame(left, bg=C_BG)
        bottom.pack(fill=tk.X, pady=(10, 0))
        self.status_label = tk.Label(bottom, text="", fg=C_INK_DIM, bg=C_BG,
                                     font=self.text_font)
        self.status_label.pack(side=tk.LEFT)
        tk.Label(bottom, text="Nida Yasmeen", fg=C_GOLD, bg=C_BG,
                 font=tkfont.Font(family="Helvetica", size=10, weight="bold", slant="italic")
                 ).pack(side=tk.RIGHT)

        # Right: info panel
        right = tk.Frame(container, bg=C_BG, width=280)
        right.pack(side=tk.LEFT, fill=tk.Y)
        right.pack_propagate(False)

        self._build_captured(right)
        self._build_history(right)
        self._build_controls(right)

    def _build_captured(self, parent):
        card = tk.Frame(parent, bg=C_PANEL, highlightbackground=C_LINE, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 14))
        tk.Label(card, text="CAPTURED", fg=C_INK_DIM, bg=C_PANEL,
                 font=self.small_font).pack(anchor="w", padx=14, pady=(12, 6))
        self.mat_label = tk.Label(card, text="material even", fg=C_INK_FAINT, bg=C_PANEL,
                                  font=self.small_font)
        self.mat_label.pack(anchor="w", padx=14)

        row1 = tk.Frame(card, bg=C_PANEL)
        row1.pack(fill=tk.X, padx=14, pady=(8, 4))
        tk.Label(row1, text="White", fg=C_INK_FAINT, bg=C_PANEL,
                 font=self.small_font, width=6).pack(side=tk.LEFT)
        self.cap_w_label = tk.Label(row1, text="", fg="#fbf6ec", bg=C_PANEL,
                                    font=tkfont.Font(family="Segoe UI Symbol", size=20))
        self.cap_w_label.pack(side=tk.LEFT)
        self.score_w_label = tk.Label(row1, text="", fg=C_AMBER, bg=C_PANEL,
                                      font=self.label_font)
        self.score_w_label.pack(side=tk.RIGHT)

        row2 = tk.Frame(card, bg=C_PANEL)
        row2.pack(fill=tk.X, padx=14, pady=(4, 14))
        tk.Label(row2, text="Black", fg=C_INK_FAINT, bg=C_PANEL,
                 font=self.small_font, width=6).pack(side=tk.LEFT)
        self.cap_b_label = tk.Label(row2, text="", fg="#2a1c12", bg=C_PANEL,
                                    font=tkfont.Font(family="Segoe UI Symbol", size=20))
        self.cap_b_label.pack(side=tk.LEFT)
        self.score_b_label = tk.Label(row2, text="", fg=C_AMBER, bg=C_PANEL,
                                      font=self.label_font)
        self.score_b_label.pack(side=tk.RIGHT)

    def _build_history(self, parent):
        card = tk.Frame(parent, bg=C_PANEL, highlightbackground=C_LINE, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, pady=(0, 14))
        tk.Label(card, text="MOVE RECORD", fg=C_INK_DIM, bg=C_PANEL,
                 font=self.small_font).pack(anchor="w", padx=14, pady=(12, 6))
        list_wrap = tk.Frame(card, bg=C_PANEL)
        list_wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 12))

        self.history_text = tk.Text(list_wrap, bg=C_PANEL2, fg=C_INK, insertbackground=C_INK,
                                    font=self.mono_font, relief="flat", padx=8, pady=6,
                                    highlightthickness=0, wrap="none", cursor="arrow",
                                    state="disabled")
        scroll = tk.Scrollbar(list_wrap, command=self.history_text.yview,
                              bg=C_PANEL, troughcolor=C_PANEL2, highlightthickness=0,
                              activebackground=C_GOLD, width=8)
        self.history_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _build_controls(self, parent):
        card = tk.Frame(parent, bg=C_PANEL, highlightbackground=C_LINE, highlightthickness=1)
        card.pack(fill=tk.X, pady=(0, 0))
        inner = tk.Frame(card, bg=C_PANEL)
        inner.pack(padx=14, pady=14, fill=tk.X)

        self.undo_btn = self._make_btn(inner, "Undo", self._do_undo)
        self.undo_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.flip_btn = self._make_btn(inner, "Flip", self._do_flip)
        self.flip_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.reset_btn = self._make_btn(inner, "New Game", self._do_reset, primary=True)
        self.reset_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

    def _make_btn(self, parent, text, command, primary=False):
        bg = C_GOLD if primary else C_PANEL2
        fg = "#2a1c10" if primary else C_INK
        btn = tk.Button(parent, text=text, command=command, textvariable=None,
                        bg=bg, fg=fg, font=self.label_font, relief="flat",
                        activebackground=C_GOLD_B if primary else C_PANEL,
                        activeforeground=fg, padx=10, pady=8, cursor="hand2",
                        highlightthickness=1, highlightbackground=C_LINE if not primary else C_GOLD,
                        highlightcolor=C_GOLD, borderwidth=0)
        return btn

    # ---- Start screen / mode selection ------------------------------------
    def _show_start_screen(self):
        win = tk.Toplevel(self.root)
        win.title("Gambit — Start")
        win.configure(bg=C_BG)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        # Prevent closing without choosing
        win.protocol("WM_DELETE_WINDOW", lambda: (win.destroy(), self.root.destroy()))

        content = tk.Frame(win, bg=C_BG, padx=40, pady=36)
        content.pack()

        # Logo
        logo = tk.Canvas(content, width=72, height=72, bg=C_BG, highlightthickness=0)
        logo.pack(pady=(0, 8))
        logo.create_oval(4, 4, 68, 68, fill=C_GOLD, outline=C_AMBER, width=3)
        logo.create_oval(10, 10, 62, 62, fill="", outline=C_GOLD_B, width=1)
        logo.create_text(36, 38, text="\u265a", fill="#2a1c10",
                         font=tkfont.Font(family=self.piece_font.actual()["family"], size=40))

        tk.Label(content, text="Gambit", fg=C_INK, bg=C_BG,
                 font=tkfont.Font(family="Helvetica", size=28, weight="bold")
                 ).pack(pady=(4, 0))
        tk.Label(content, text="CLASSICAL CHESS", fg=C_INK_FAINT, bg=C_BG,
                 font=tkfont.Font(family="Helvetica", size=11, weight="bold")
                 ).pack(pady=(0, 24))

        tk.Label(content, text="Choose Playing Mode", fg=C_GOLD_B, bg=C_BG,
                 font=self.label_font).pack(pady=(0, 14))

        mode_var = tk.StringVar(value="human")
        color_var = tk.StringVar(value="white")
        depth_var = tk.IntVar(value=3)

        # AI options panel (color + difficulty) — created first so update_options can reference
        options_frame = tk.Frame(content, bg=C_BG)
        options_frame.pack(fill=tk.X, pady=(14, 0))

        color_frame = tk.Frame(options_frame, bg=C_BG)
        color_frame.pack(fill=tk.X)
        tk.Label(color_frame, text="You play as:", fg=C_INK_DIM, bg=C_BG,
                 font=self.small_font).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(color_frame, text="White", variable=color_var, value="white",
                       bg=C_BG, fg=C_INK, selectcolor=C_BG, activebackground=C_BG,
                       font=self.text_font, highlightthickness=0, borderwidth=0
                       ).pack(side=tk.LEFT, padx=(0, 12))
        tk.Radiobutton(color_frame, text="Black", variable=color_var, value="black",
                       bg=C_BG, fg=C_INK, selectcolor=C_BG, activebackground=C_BG,
                       font=self.text_font, highlightthickness=0, borderwidth=0
                       ).pack(side=tk.LEFT)

        diff_frame = tk.Frame(options_frame, bg=C_BG)
        diff_frame.pack(fill=tk.X, pady=(8, 0))
        tk.Label(diff_frame, text="Difficulty:", fg=C_INK_DIM, bg=C_BG,
                 font=self.small_font).pack(side=tk.LEFT, padx=(0, 10))
        for label, val in [("Easy", 2), ("Medium", 3), ("Hard", 4)]:
            tk.Radiobutton(diff_frame, text=label, variable=depth_var, value=val,
                           bg=C_BG, fg=C_INK, selectcolor=C_BG, activebackground=C_BG,
                           font=self.text_font, highlightthickness=0, borderwidth=0
                           ).pack(side=tk.LEFT, padx=(0, 12))

        def update_options():
            is_ai = mode_var.get() == "ai"
            state = "normal" if is_ai else "disabled"
            for child in color_frame.winfo_children() + diff_frame.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass

        def make_mode(text, value, desc):
            f = tk.Frame(content, bg=C_PANEL2, highlightbackground=C_LINE, highlightthickness=2,
                         cursor="hand2")
            f.pack(fill=tk.X, pady=5, ipady=8)
            rb = tk.Radiobutton(f, text=text, variable=mode_var, value=value,
                                bg=C_PANEL2, fg=C_INK, selectcolor=C_PANEL2,
                                activebackground=C_PANEL2, activeforeground=C_INK,
                                font=self.label_font, highlightthickness=0, borderwidth=0,
                                command=update_options)
            rb.pack(side=tk.LEFT, padx=(14, 8))
            tk.Label(f, text=desc, fg=C_INK_FAINT, bg=C_PANEL2,
                     font=self.small_font).pack(side=tk.LEFT)
            # Click anywhere on the frame selects the radio button
            f.bind("<Button-1>", lambda e: (mode_var.set(value), update_options()))

        make_mode("Human vs Human", "human", "Two players, one screen")
        make_mode("Human vs Computer", "ai", "Challenge the computer")

        update_options()

        def start():
            is_ai = mode_var.get() == "ai"
            self.ai_opponent = is_ai
            if is_ai:
                self.ai_color = Color.BLACK if color_var.get() == "white" else Color.WHITE
                self.ai_depth = depth_var.get()
                self.ai = ChessAI(self.ai_depth)
                self.flipped = (self.ai_color is Color.WHITE)
            else:
                self.ai_color = Color.BLACK
                self.ai = None
                self.flipped = False
            win.destroy()
            self.refresh()
            self._trigger_ai()

        tk.Button(content, text="Start Game", command=start,
                  bg=C_GOLD, fg="#2a1c10", font=self.label_font, relief="flat",
                  padx=32, pady=10, cursor="hand2", borderwidth=0,
                  activebackground=C_GOLD_B).pack(pady=(24, 0))

        win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    # ---- AI integration ---------------------------------------------------
    def _trigger_ai(self):
        """If it's the AI's turn, compute and play its move in a thread."""
        if not self.ai_opponent or self.game.game_over or self.pending_promo:
            return
        if self.game.board.turn is not self.ai_color:
            return
        if self.ai_thinking:
            return

        self.ai_thinking = True
        self.status_label.configure(text="Computer is thinking...", fg=C_AMBER)

        def compute():
            move = self.ai.best_move(self.game)
            self.root.after(0, lambda: self._ai_move_done(move))

        self.ai_thread = threading.Thread(target=compute, daemon=True)
        self.ai_thread.start()

    def _ai_move_done(self, move):
        self.ai_thinking = False
        if move is None or self.game.game_over:
            self.refresh()
            return
        self.game.make_move(move)
        self.selected = None
        self.legal_targets = []
        self.refresh()
        self._check_game_over()

    # ---- Coordinate helpers -----------------------------------------------
    def _sq_to_pixel(self, r, c):
        if self.flipped:
            x = (7 - c) * self.SQ
            y = (7 - r) * self.SQ
        else:
            x = c * self.SQ
            y = r * self.SQ
        return x, y

    def _pixel_to_sq(self, px, py):
        c = int(px // self.SQ)
        r = int(py // self.SQ)
        if self.flipped:
            c = 7 - c
            r = 7 - r
        if 0 <= r < 8 and 0 <= c < 8:
            return r, c
        return None

    # ---- Rendering --------------------------------------------------------
    def refresh(self):
        self._draw_board()
        self._update_turn()
        self._update_status()
        self._update_captured()
        self._update_history()
        self.undo_btn.configure(state="normal" if self.game.history else "disabled")

    def _draw_board(self):
        c = self.canvas
        c.delete("all")
        board = self.game.board
        check_sq = None
        if not self.game.game_over and board.in_check(board.turn):
            check_sq = board.find_king(board.turn)

        for r in range(8):
            for col in range(8):
                x, y = self._sq_to_pixel(r, col)
                is_light = (r + col) % 2 == 0
                base = C_LSQ if is_light else C_DSQ

                # Last-move highlight
                if self.game.history:
                    lm = self.game.history[-1].move
                    if (r, col) == lm.from_sq or (r, col) == lm.to_sq:
                        base = C_LSQ_HI if is_light else C_DSQ_HI

                # Check highlight
                if check_sq and (r, col) == check_sq:
                    c.create_rectangle(x, y, x + self.SQ, y + self.SQ,
                                       fill=C_CHECK, outline="", tags="sq")
                else:
                    c.create_rectangle(x, y, x + self.SQ, y + self.SQ,
                                       fill=base, outline="", tags="sq")

                # Selected highlight
                if self.selected and self.selected == (r, col):
                    c.create_rectangle(x + 2, y + 2, x + self.SQ - 2, y + self.SQ - 2,
                                       outline=C_SEL, width=3, tags="sel")

                # Coordinates
                bottom_row = 7 if not self.flipped else 0
                left_col = 0 if not self.flipped else 7
                if r == bottom_row:
                    c.create_text(x + self.SQ - 8, y + self.SQ - 8,
                                  text=FILES_LIST[col], fill=C_INK_FAINT,
                                  font=self.coord_font, tags="coord")
                if col == left_col:
                    c.create_text(x + 8, y + 8, text=str(8 - r),
                                  fill=C_INK_FAINT, font=self.coord_font, tags="coord")

                # Piece
                p = board.squares[r][col]
                if p:
                    glyph = PIECE_GLYPH[p.fen_char]
                    color = "#fbf6ec" if p.is_white else "#2a1c12"
                    c.create_text(x + self.SQ // 2, y + self.SQ // 2,
                                  text=glyph, fill=color, font=self.piece_font, tags="piece")

                # Legal move indicators
                if (r, col) in self.legal_targets:
                    if p or (board.en_passant and (r, col) == board.en_passant):
                        c.create_oval(x + 6, y + 6, x + self.SQ - 6, y + self.SQ - 6,
                                      outline="rgba(40,30,20,0)", width=3, tags="hint")
                        c.create_oval(x + 6, y + 6, x + self.SQ - 6, y + self.SQ - 6,
                                      outline="#5a4030", width=3, tags="hint")
                    else:
                        cx, cy = x + self.SQ // 2, y + self.SQ // 2
                        c.create_oval(cx - 8, cy - 8, cx + 8, cy + 8,
                                      fill="#5a4030", outline="", tags="hint")

    def _update_turn(self):
        c = self.turn_dot
        c.delete("all")
        if self.game.game_over:
            self.turn_label.configure(text="Game over")
            c.create_oval(3, 3, 13, 13, fill=C_INK_FAINT, outline="")
        else:
            white = self.game.board.turn is Color.WHITE
            self.turn_label.configure(text=f"{'White' if white else 'Black'} to move")
            if white:
                c.create_oval(3, 3, 13, 13, fill="#ffffff", outline=C_GOLD)
            else:
                c.create_oval(3, 3, 13, 13, fill="#1a1410", outline=C_GOLD)

    def _update_status(self):
        if self.ai_thinking:
            return  # keep "Computer is thinking..." message
        if self.game.game_over:
            self.status_label.configure(text="")
        elif self.game.board.in_check(self.game.board.turn):
            side = "White" if self.game.board.turn is Color.WHITE else "Black"
            self.status_label.configure(text=f"{side} is in check", fg=C_CRIMSON)
        else:
            self.status_label.configure(text="")

    def _update_captured(self):
        cap_w, cap_b = [], []
        for entry in self.game.history:
            if not entry.captured:
                continue
            if entry.captured.color is Color.BLACK:
                cap_w.append(entry.captured.fen_char)
            else:
                cap_b.append(entry.captured.fen_char)
        cap_w.sort(key=lambda p: -PIECE_VALUE[p.lower()])
        cap_b.sort(key=lambda p: -PIECE_VALUE[p.lower()])

        self.cap_w_label.configure(text="".join(PIECE_GLYPH[p] for p in cap_w))
        self.cap_b_label.configure(text="".join(PIECE_GLYPH[p] for p in cap_b))

        sw = sum(PIECE_VALUE[p.lower()] for p in cap_w)
        sb = sum(PIECE_VALUE[p.lower()] for p in cap_b)
        diff = sw - sb
        self.score_w_label.configure(text=f"+{diff}" if diff > 0 else "")
        self.score_b_label.configure(text=f"+{-diff}" if diff < 0 else "")
        if diff == 0:
            self.mat_label.configure(text="material even")
        elif diff > 0:
            self.mat_label.configure(text=f"white +{diff}")
        else:
            self.mat_label.configure(text=f"black +{-diff}")

    def _update_history(self):
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", tk.END)
        if not self.game.history:
            self.history_text.insert("1.0", "  The board is set. White begins.")
            self.history_text.configure(state="disabled")
            return
        for i in range(0, len(self.game.history), 2):
            num = i // 2 + 1
            w_move = self.game.history[i].san
            b_move = self.game.history[i + 1].san if i + 1 < len(self.game.history) else ""
            line = f"  {num:>2}.  {w_move:<8}  {b_move}\n"
            self.history_text.insert(tk.END, line)
        self.history_text.see(tk.END)
        self.history_text.configure(state="disabled")

    # ---- Interaction ------------------------------------------------------
    def _on_click(self, event):
        if self.game.game_over or self.pending_promo or self.ai_thinking:
            return
        # Block clicks on AI's pieces
        sq = self._pixel_to_sq(event.x, event.y)
        if not sq:
            return
        r, c = sq
        board = self.game.board
        piece = board.squares[r][c]
        if self.ai_opponent and piece and piece.color is self.ai_color and not self.selected:
            return

        if self.selected:
            sr, sc = self.selected
            for mv in board.legal_moves_from(sr, sc):
                if mv.to_sq == (r, c):
                    if mv.promotion:
                        self.pending_promo = mv
                        self._show_promo_dialog(mv.piece.color)
                        return
                    self.game.make_move(mv)
                    self.selected = None
                    self.legal_targets = []
                    self.refresh()
                    self._check_game_over()
                    self._trigger_ai()
                    return
            if self.selected == (r, c):
                self.selected = None
                self.legal_targets = []
                self.refresh()
                return

        if piece and piece.color is board.turn:
            self.selected = (r, c)
            self.legal_targets = [mv.to_sq for mv in board.legal_moves_from(r, c)]
        else:
            self.selected = None
            self.legal_targets = []
        self.refresh()

    def _on_key(self, event):
        key = event.char.lower()
        if key == "u":
            self._do_undo()
        elif key == "r":
            self._do_reset()
        elif key == "f":
            self._do_flip()

    def _do_undo(self):
        if self.ai_thinking:
            return
        self.game.undo()
        # In AI mode, undo the AI's move too so it's the human's turn again
        if self.ai_opponent and self.game.history and self.game.board.turn is self.ai_color:
            self.game.undo()
        self.selected = None
        self.legal_targets = []
        self.pending_promo = None
        self.refresh()

    def _do_reset(self):
        self.game = Game()
        self.selected = None
        self.legal_targets = []
        self.pending_promo = None
        self.ai_thinking = False
        self._show_start_screen()

    def _do_reset(self):
        self.game = Game()
        self.selected = None
        self.legal_targets = []
        self.pending_promo = None
        self.refresh()

    def _do_flip(self):
        self.flipped = not self.flipped
        self.refresh()

    # ---- Promotion dialog -------------------------------------------------
    def _show_promo_dialog(self, color):
        win = tk.Toplevel(self.root)
        win.title("Promote Pawn")
        win.configure(bg=C_PANEL)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        tk.Label(win, text="Choose promotion piece", fg=C_GOLD_B, bg=C_PANEL,
                 font=self.label_font).pack(padx=24, pady=(20, 12))

        choice_frame = tk.Frame(win, bg=C_PANEL)
        choice_frame.pack(padx=24, pady=(0, 20))

        white = color is Color.WHITE
        for ptype in (PieceType.QUEEN, PieceType.ROOK, PieceType.BISHOP, PieceType.KNIGHT):
            ch = ptype.value.upper() if white else ptype.value
            glyph = PIECE_GLYPH[ch]
            btn = tk.Button(choice_frame, text=glyph,
                            font=tkfont.Font(family=self.piece_font.actual()["family"], size=32),
                            bg=C_PANEL2, fg="#fbf6ec" if white else "#2a1c12",
                            activebackground=C_GOLD, relief="flat", padx=16, pady=12,
                            cursor="hand2", highlightthickness=1, highlightbackground=C_LINE,
                            borderwidth=0,
                            command=lambda pt=ptype: self._choose_promo(pt, win))
            btn.pack(side=tk.LEFT, padx=5)

        win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    def _choose_promo(self, ptype, win):
        win.destroy()
        mv = self.pending_promo
        self.pending_promo = None
        mv.promotion = ptype
        self.game.make_move(mv)
        self.selected = None
        self.legal_targets = []
        self.refresh()
        self._check_game_over()
        self._trigger_ai()

    # ---- Game over dialog -------------------------------------------------
    def _check_game_over(self):
        if not self.game.game_over:
            return
        win = tk.Toplevel(self.root)
        win.title("Game Over")
        win.configure(bg=C_PANEL)
        win.resizable(False, False)
        win.transient(self.root)
        win.grab_set()

        res = self.game.result
        if res == "1-0":
            crown, title, msg = "\u2654", "Checkmate", "White wins by checkmate."
        elif res == "0-1":
            crown, title, msg = "\u265a", "Checkmate", "Black wins by checkmate."
        else:
            crown = "\u00bd"
            if self.game.board.is_stalemate():
                title, msg = "Stalemate", "The game is drawn \u2014 no legal moves remain."
            else:
                title, msg = "Draw", "The game is drawn."

        tk.Label(win, text=crown, fg=C_GOLD, bg=C_PANEL,
                 font=tkfont.Font(family="Segoe UI Symbol", size=48)).pack(pady=(28, 4))
        tk.Label(win, text=title, fg=C_INK, bg=C_PANEL,
                 font=tkfont.Font(family="Helvetica", size=22, weight="bold")).pack(pady=(0, 6))
        tk.Label(win, text=msg, fg=C_INK_DIM, bg=C_PANEL,
                 font=self.text_font).pack(pady=(0, 20))

        tk.Button(win, text="Play Again", command=lambda: (win.destroy(), self._do_reset()),
                  bg=C_GOLD, fg="#2a1c10", font=self.label_font, relief="flat",
                  padx=28, pady=10, cursor="hand2", borderwidth=0,
                  activebackground=C_GOLD_B).pack(pady=(0, 24))

        win.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{x}+{y}")

    # ---- Run --------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        print("""
Gambit \u2014 Chess

Usage:
  python chess.py          Launch the GUI (default)
  python chess.py --cli    Play in the terminal
  python chess.py --help   Show this help
""".strip())
        sys.exit(0)
    if "--cli" in args:
        play_cli()
    else:
        ChessGUI().run()