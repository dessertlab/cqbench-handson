from __future__ import annotations

import ast
import io
import keyword
import re
import tokenize
from dataclasses import dataclass, asdict
from typing import Any, Iterable

import lizard


STUB_WORD_RE = re.compile(r"\b(?:todo|fixme|implementation\s+goes\s+here)\b", re.I)
NOT_IMPLEMENTED_RE = re.compile(
    r"(?:NotImplementedError|UnsupportedOperationException|not\s+implemented)", re.I
)


@dataclass(frozen=True)
class Signature:
    text: str
    name: str
    arity: int | None


@dataclass(frozen=True)
class StructuralResult:
    parseable: bool
    target_present: bool
    target_matches_arity: bool
    explicit_stub: bool
    constant_noop_only: bool
    undersized: bool
    nonstub: bool
    strict_nontrivial: bool
    status: str
    token_count: int
    ast_node_count: int
    statement_count: int
    target_token_ratio: float | None
    target_ast_ratio: float | None
    diagnostics: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _lizard_functions(code: str, language: str):
    filename = {"python": "temp.py", "java": "Temp.java", "c": "temp.c"}[language]
    return list(lizard.analyze_file.analyze_source_code(filename, code).function_list)


def extract_signature(code: str, language: str) -> Signature:
    if language == "python":
        tree = ast.parse(code)
        nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert nodes, "No Python function in human reference"
        node = min(nodes, key=lambda item: (item.lineno, item.col_offset))
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        segment = ast.get_source_segment(code, node) or ""
        header = segment.split(":", 1)[0].strip()
        if not header.startswith(prefix):
            header = f"{prefix} {node.name}(...)"
        arity = len(node.args.posonlyargs) + len(node.args.args) + len(node.args.kwonlyargs)
        return Signature(header, node.name, arity)

    functions = _lizard_functions(code, language)
    assert functions, f"No {language} function in human reference"
    function = min(functions, key=lambda item: (int(item.start_line), int(item.end_line)))
    lines = code.splitlines()
    start = max(0, int(function.start_line) - 1)
    fragment = "\n".join(lines[start:])
    brace = fragment.find("{")
    assert brace >= 0, f"Cannot locate signature brace for {function.name}"
    text = re.sub(r"\s+", " ", fragment[:brace]).strip()
    name = str(function.name).rsplit("::", 1)[-1].rsplit(".", 1)[-1]
    return Signature(text, name, int(function.parameter_count))


def canonical_prompt(language: str, signature: Signature, docstring: str) -> str:
    return (
        f"Implement the following {language} function. Return code only. "
        "Preserve the required function name and parameters.\n\n"
        f"Signature:\n{signature.text}\n\nSpecification:\n{docstring.strip()}"
    )


def _python_tokens(code: str) -> int:
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        ignored = {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }
        return sum(token.type not in ignored for token in stream)
    except (IndentationError, tokenize.TokenError):
        return len(re.findall(r"\w+|[^\w\s]", code))


def _generic_tokens(code: str) -> int:
    return len(re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||->|[^\s]", code))


def _tree_sitter(language: str, code: str):
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError as exc:
        raise RuntimeError(
            "tree_sitter_language_pack is required for Java/C structural validation"
        ) from exc
    parser = get_parser(language)
    return parser.parse(code.encode("utf-8"))


def _walk(node) -> Iterable[Any]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _node_text(code_bytes: bytes, node) -> str:
    return code_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _python_target(code: str, signature: Signature):
    tree = ast.parse(code)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == signature.name
    ]
    if not matches:
        return tree, None, False
    exact = [
        node
        for node in matches
        if (
            len(node.args.posonlyargs)
            + len(node.args.args)
            + len(node.args.kwonlyargs)
        )
        == signature.arity
    ]
    return tree, (exact or matches)[0], bool(exact)


def _python_statement_count(node) -> int:
    if node is None:
        return 0
    return sum(isinstance(item, ast.stmt) for item in ast.walk(node)) - 1


def _python_stub(node, code: str) -> tuple[bool, bool]:
    if node is None:
        return False, False
    body = list(node.body)
    if len(body) == 1:
        item = body[0]
        if isinstance(item, ast.Pass):
            return True, False
        if isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
            if item.value.value is Ellipsis:
                return True, False
            if isinstance(item.value.value, str) and STUB_WORD_RE.search(item.value.value):
                return True, False
        if isinstance(item, ast.Raise) and NOT_IMPLEMENTED_RE.search(ast.unparse(item)):
            return True, False
        if isinstance(item, ast.Return) and (
            item.value is None or isinstance(item.value, ast.Constant)
        ):
            return False, True
    source = ast.get_source_segment(code, node) or ""
    return bool(STUB_WORD_RE.search(source) and len(body) <= 2), False


def _generic_target(language: str, code: str, signature: Signature):
    tree = _tree_sitter(language, code)
    code_bytes = code.encode("utf-8")
    error = any(node.type == "ERROR" for node in _walk(tree.root_node))
    function_types = {"function_definition", "method_declaration", "constructor_declaration"}
    candidates = []
    for node in _walk(tree.root_node):
        if node.type not in function_types:
            continue
        text = _node_text(code_bytes, node)
        if re.search(rf"\b{re.escape(signature.name)}\s*\(", text):
            candidates.append((node, text))
    if not candidates:
        return tree, None, False, error, ""
    exact = []
    for node, text in candidates:
        prefix = text.split("{", 1)[0]
        match = re.search(r"\((.*)\)", prefix, flags=re.S)
        params = match.group(1).strip() if match else ""
        arity = 0 if not params or params == "void" else params.count(",") + 1
        if signature.arity is None or arity == signature.arity:
            exact.append((node, text))
    chosen = (exact or candidates)[0]
    return tree, chosen[0], bool(exact), error, chosen[1]


def _generic_counts(node) -> tuple[int, int]:
    if node is None:
        return 0, 0
    nodes = list(_walk(node))
    statement_suffixes = ("statement", "declaration")
    statements = sum(
        item.type.endswith(statement_suffixes)
        or item.type in {"return_statement", "expression_statement"}
        for item in nodes
    )
    return len(nodes), statements


def _generic_stub(text: str) -> tuple[bool, bool]:
    if not text:
        return False, False
    body_match = re.search(r"\{(.*)\}\s*$", text, flags=re.S)
    body = body_match.group(1).strip() if body_match else ""
    stripped = re.sub(r"//.*?$|/\*.*?\*/", "", body, flags=re.M | re.S).strip()
    if not stripped or STUB_WORD_RE.search(stripped) or NOT_IMPLEMENTED_RE.search(stripped):
        return True, False
    constant = re.fullmatch(
        r"return\s+(?:null|NULL|nullptr|true|false|-?\d+(?:\.\d+)?|\"[^\"]*\")\s*;",
        stripped,
    )
    return False, bool(constant)


def analyze_structure(
    code: str,
    language: str,
    signature: Signature,
    *,
    human_token_count: int | None = None,
    human_ast_count: int | None = None,
) -> StructuralResult:
    diagnostics: list[str] = []
    code = code or ""
    if not code.strip():
        return StructuralResult(
            False, False, False, True, False, True, False, False,
            "empty", 0, 0, 0, 0.0, 0.0, ("empty output",),
        )
    token_count = _python_tokens(code) if language == "python" else _generic_tokens(code)
    try:
        if language == "python":
            tree, target, arity_ok = _python_target(code, signature)
            parseable = True
            target_present = target is not None
            ast_count = sum(1 for _ in ast.walk(target)) if target is not None else 0
            statements = _python_statement_count(target)
            explicit_stub, constant = _python_stub(target, code)
        else:
            tree, target, arity_ok, has_error, text = _generic_target(
                language, code, signature
            )
            parseable = not has_error
            target_present = target is not None
            ast_count, statements = _generic_counts(target)
            explicit_stub, constant = _generic_stub(text)
            if has_error:
                diagnostics.append("tree-sitter parse contains ERROR nodes")
    except Exception as exc:
        return StructuralResult(
            False, False, False, False, False, False, False, False,
            "parse_error", token_count, 0, 0, None, None, (str(exc),),
        )

    token_ratio = (
        token_count / human_token_count
        if human_token_count is not None and human_token_count > 0
        else None
    )
    ast_ratio = (
        ast_count / human_ast_count
        if human_ast_count is not None and human_ast_count > 0
        else None
    )
    undersized = bool(
        statements <= 2
        and token_ratio is not None
        and ast_ratio is not None
        and token_ratio < 0.10
        and ast_ratio < 0.10
    )
    nonstub = bool(parseable and target_present and arity_ok and not explicit_stub)
    strict = bool(nonstub and not constant and not undersized)
    if not parseable:
        status = "parse_error"
    elif not target_present:
        status = "target_missing"
    elif not arity_ok:
        status = "arity_mismatch"
    elif explicit_stub:
        status = "explicit_stub"
    elif constant:
        status = "constant_noop"
    elif undersized:
        status = "undersized"
    else:
        status = "nontrivial"
    return StructuralResult(
        parseable,
        target_present,
        arity_ok,
        explicit_stub,
        constant,
        undersized,
        nonstub,
        strict,
        status,
        token_count,
        ast_count,
        statements,
        token_ratio,
        ast_ratio,
        tuple(diagnostics),
    )

