"""User-facing tools and an example external service."""

from __future__ import annotations

import ast
import math
import operator
import urllib.error
import urllib.request
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Union

from .._version import USER_AGENT
from ..core._util import cs_str, is_blank
from ..core.abstractions import ICapability
from ..core.model.capability import CapabilityRequest, CapabilityResult
from ..core.model.data import ListData, TextData

Number = Union[int, float, Decimal]


class EchoTool(ICapability):
    """Returns its input unchanged; fails on empty input."""

    name = "echo"

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        if is_blank(request.input):
            return CapabilityResult.fail(self.name, "Input was empty")
        return CapabilityResult.ok(self.name, TextData(value=request.input))


class SearchTool(ICapability):
    """Stub search capability that returns two placeholder results for the query."""

    name = "search"

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult.ok(self.name, ListData(rows=[
            f"Result 1 for '{request.input}'",
            f"Result 2 for '{request.input}'",
        ]))


class FailingTool(ICapability):
    """Test capability that always raises, for exercising failure handling. Not registered by default."""

    name = "fail"

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        raise RuntimeError("Simulated tool failure")


class CalculationError(ValueError):
    """An expression the calculator cannot evaluate (the equivalent of ``DataTable.Compute`` throwing)."""


class CalculatorTool(ICapability):
    """Evaluates arithmetic expressions (e.g. ``calc 10+20``).

    The C# version uses ``DataTable.Compute``; this port uses a small, safe evaluator with the same arithmetic
    semantics: integer literals stay integers for ``+ - * %``, decimal literals are exact decimals (``0.1+0.2`` is
    ``0.3``), and dividing integers produces a floating-point result (``7/2`` is ``3.5``). Only numbers, parentheses,
    unary ``+``/``-`` and ``+ - * / %`` are accepted. Invalid expressions raise; ``CapabilityExecutor`` converts the
    exception into a failed result, which execution invariants then reject.
    """

    name = "calc"

    _BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: None, ast.Mod: None}

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        return CapabilityResult.ok(self.name, TextData(value=_render(self.compute(request.input))))

    def compute(self, expression: str) -> Number:
        """Evaluates ``expression``; raises :class:`ValueError` when it is not valid arithmetic."""
        if is_blank(expression):
            raise CalculationError("Syntax error: the expression is empty.")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as ex:
            raise CalculationError(f"Syntax error in the expression '{expression}'.") from ex
        return self._eval(tree.body, expression.strip())

    def _eval(self, node: ast.AST, source: str) -> Number:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            if isinstance(node.value, float):
                return Decimal(ast.get_source_segment(source, node) or repr(node.value))
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval(node.operand, source)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in self._BINARY:
            left, right = self._eval(node.left, source), self._eval(node.right, source)
            if isinstance(left, Decimal) or isinstance(right, Decimal):
                left, right = _to_decimal(left), _to_decimal(right)
            try:
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        raise CalculationError("Attempted to divide by zero.")
                    return left / right if isinstance(left, Decimal) else float(left) / float(right)
                if isinstance(node.op, ast.Mod):
                    if right == 0:
                        raise CalculationError("Attempted to divide by zero.")
                    return _cs_mod(left, right)
                return self._BINARY[type(node.op)](left, right)
            except (DivisionByZero, InvalidOperation, ZeroDivisionError) as ex:
                raise CalculationError(str(ex)) from ex
        raise CalculationError(f"Syntax error in the expression '{source}'.")


def _to_decimal(value: Number) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(repr(value) if isinstance(value, float) else value)


def _cs_mod(left: Number, right: Number) -> Number:
    # C# % truncates towards zero (the result takes the sign of the dividend), unlike Python's floored %.
    if isinstance(left, Decimal):
        return left % right  # Decimal % already truncates towards zero
    if isinstance(left, float) or isinstance(right, float):
        return math.fmod(left, right)
    result = abs(left) % abs(right)
    return -result if left < 0 else result


def _render(value: Number) -> str:
    if isinstance(value, Decimal):
        return str(value)
    return cs_str(value)


class ExampleHttpService(ICapability):
    """Example external-service capability that performs an HTTP GET against a fixed URL.

    Demonstrates wrapping an external service as a capability.
    """

    name = "http_example"
    #: Fixed URL, as in the reference implementation.
    URL = "http://example.com"

    def execute(self, request: CapabilityRequest) -> CapabilityResult:
        try:
            req = urllib.request.Request(self.URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as response:  # noqa: S310 - fixed URL
                content = response.read().decode("utf-8", errors="replace")
                return CapabilityResult(capability=self.name, success=200 <= response.status < 300,
                                        data=TextData(value=content))
        except urllib.error.HTTPError as ex:
            return CapabilityResult(capability=self.name, success=False, data=TextData(value=""), error=str(ex))
        except Exception as ex:  # noqa: BLE001
            return CapabilityResult(capability=self.name, success=False, error=str(ex))
