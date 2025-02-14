import typing
from typing import Union, Any
from collections import defaultdict

from .semanticanalysis import IfWrapper, SemanticAnalysisError, \
    find_use, find_uses, find_all, is_return, is_void_return, \
    Scope, ClassDefinition, InterfaceDefinition, creates_new_variable_scope, VariableDefinition, \
    LocalVariableDefinition, GlobalVariableDefinition, ParameterDefinition, type_node_to_list_of_types, \
    list_to_typed_node, list_to_attribute_access_node, is_call_lambda, \
    nested_same_binop_to_list, gensym, FieldDefinition
from .abstractsyntaxtree import Node, Module, Call, Block, UnOp, BinOp, TypeOp, Assign, Identifier, ListLiteral, TupleLiteral, BracedLiteral, ArrayAccess, BracedCall, StringLiteral, AttributeAccess, Template, ArrowOp, ScopeResolution, LeftAssociativeUnOp, IntegerLiteral, FloatLiteral, NamedParameter, SyntaxTypeOp

from collections import defaultdict
import re
import textwrap


mut_by_default = False


class CodeGenError(Exception):
    pass


cpp_preamble = r"""
#include <string>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <functional>
#include <cassert>
#include <compare> // for <=>
#include <thread>
#include <optional>


#include "ceto.h"

"""


def codegen(expr: Node):
    assert isinstance(expr, Module)
    cx = Scope()
    s = codegen_module(expr, cx)
    print(s)
    return s


def codegen_module(module: Module, cx: Scope):
    assert isinstance(module, Module)
    modcpp = ""

    included_module_code = defaultdict(str)

    for modarg in module.args:

        if isinstance(modarg, Call) and modarg.func.name == "def":
            funcx = cx.enter_scope()
            funcx.in_function_param_list = True
            modarg_code = codegen_def(modarg, funcx)
        else:
            modarg_code = codegen_block_item(modarg, cx)

        if modarg.source.header_file_h:
            included_module_code[modarg.source.header_file_h] += modarg_code
        else:
            modcpp += modarg_code

    for path, include_code in included_module_code.items():
        with open(path, "w") as include_file:
            include_file.write("#pragma once\n" + cpp_preamble + include_code)

    return cpp_preamble + modcpp


def codegen_block(block: Block, cx):
    assert isinstance(block, Block)
    cpp = ""
    indent_str = cx.indent_str()

    for b in block.args:
        cpp += indent_str + codegen_block_item(b, cx)

    return cpp


def codegen_block_item(b : Node, cx):
    assert isinstance(b, Node)

    if isinstance(b, Identifier):
        if b.name == "pass":
            return "; // pass\n"

    if isinstance(b, Call):
        if b.func.name == "for":
            return codegen_for(b, cx)
        elif b.func.name in ["class", "struct"]:
            return codegen_class(b, cx)
        elif b.func.name == "if":
            return codegen_if(b, cx)
        elif b.func.name == "while":
            return codegen_while(b, cx)
        elif b.func.name == "def":
            return codegen_def(b, cx)

    if b.declared_type is not None:
        # typed declaration

        if b.declared_type.name == "global":
            if not cx.in_function_body:
                raise CodeGenError("global declarations only allowed in function bodies", b)
            if not isinstance(b, Identifier):
                raise CodeGenError("global declarations only apply to an Identifier", b)
            return ""  # default variable declaration handling in Scope already handles

        types = type_node_to_list_of_types(b.declared_type)
        if any(t.name in ["typedef", "using"] for t in types):
            # TODO more error checking here
            # we might just want to ban 'using' altogether (dangerous in combination with _ceto_ defined classes (not structs)
            declared_type = b.declared_type
            cpp = codegen_type(b, b.declared_type, cx)
            b.declared_type = None
            cpp += " " + codegen_node(b, cx) + ";\n"
            b.declared_type = declared_type
            return cpp

        field_type_const_part, field_type_str = codegen_variable_declaration_type(b, cx)
        decl = field_type_const_part + field_type_str + " " + b.name
        return " " + decl + ";\n"

    cpp = codegen_node(b, cx)
    if not is_comment(b):
        cpp += ";\n"
    return cpp


def codegen_def(defnode: Call, cx):
    assert defnode.func.name == "def"
    name_node = defnode.args[0]
    name = name_node.name
    args = defnode.args[1:]
    if args and isinstance(args[-1], Block):
        block = args.pop()
        is_declaration = False
    else:
        block = None
        is_declaration = True

    return_type_node = defnode.declared_type

    if isinstance(name_node, Call) and name_node.func.name == "operator" and len(name_node.args) == 1 and isinstance(operator_name_node := name_node.args[0], StringLiteral):
        name = "operator" + operator_name_node.str

    if name is None:
        # no immediate plans to support out of line methods
        raise CodeGenError(f"can't handle name {name_node} in def {defnode}")

    params = []
    typenames = []

    # no support for out of line methods at the moment
    class_identifier = class_name_node_from_inline_method(defnode)
    is_method = class_identifier is not None
    if is_method:
        if isinstance(class_identifier, Identifier):
            class_name = class_identifier.name
        elif isinstance(class_identifier, Template):
            class_name = class_identifier.func.name + "<" + ", ".join(codegen_node(t, cx) for t in class_identifier.args) + ">"
        else:
            assert 0, class_identifier

    else:
        class_name = None

    is_destructor = False
    if is_method and name == "destruct":
        if args:
            raise CodeGenError("destructors can't take arguments", defnode)
        is_destructor = True

    specifier_node = name_node.declared_type
    specifier_types = type_node_to_list_of_types(specifier_node)

    interface_calls = [t for t in specifier_types if isinstance(t, Call) and t.func.name == "interface"]
    if len(interface_calls) > 1:
        raise CodeGenError("too many 'interface' specifiers", defnode)
    elif len(interface_calls) == 1:
        interface = interface_calls[0]
        specifier_types.remove(interface)
    else:
        interface = None

    if interface and return_type_node is None:
        raise CodeGenError("must specify return type of interface method")
    assert is_method or not interface

    for i, arg in enumerate(args):
        if interface:
            if arg.declared_type is None:
                raise CodeGenError(
                    "parameter types must be specified for interface methods")
            if not isinstance(arg, Identifier):
                raise CodeGenError(
                    "Only simple args allowed for interface method (c++ virtual functions with default arguments are best avoided)")
        if typed_param := codegen_typed_def_param(arg, cx):
            params.append(typed_param)
        else:
            t = "T" + str(i + 1)
            # params.append(t + "&& " + arg.name)
            # params.append(t + " " + arg.name)
            params.append("const " + t + "& " + arg.name)
            typenames.append("typename " + t)

    template = ""
    inline = "inline "
    override = ""
    final = ""
    specifier = ""
    is_const = is_method and not mut_by_default

    if specifier_types:
        const_or_mut = [t for t in specifier_types if t.name in ["const", "mut"]]
        if len(const_or_mut) > 1:
            raise CodeGenError("too many 'mut' and 'const' specified", defnode)

        if const_or_mut:
            if not is_method:
                raise CodeGenError("Don't specify const/mut for a non-method", const_or_mut[0])
            is_const = const_or_mut[0].name == "const"
            specifier_types.remove(const_or_mut[0])

        overrides = [t for t in specifier_types if t.name == "override"]
        if overrides:
            if len(overrides) > 1:
                raise CodeGenError("too many overrides specified", defnode)
            if not is_method:
                raise CodeGenError("Don't specify 'override' for a non-method", defnode)
            specifier_types.remove(overrides[0])

        if overrides or interface:
            override = " override"

        finals = [t for t in specifier_types if t.name == "final"]
        if finals:
            if len(finals) > 1:
                raise CodeGenError("too many 'final' specified", defnode)
            if not is_method:
                raise CodeGenError("Don't specify 'final' for a non-method", defnode)
            specifier_types.remove(finals[0])
            final = " final"

        noinlines = [t for t in specifier_types if t.name == "noinline"]
        if noinlines:
            if len(noinlines) > 1:
                raise CodeGenError("too many 'noinline' specified", defnode)
            specifier_types.remove(noinlines[0])
            inline = ""

        # maybe it's ok to put the requires clause with the return type
        #adjacent_specifier_types = zip(specifier_types, specifier_types[1:])
        #for t, y in adjacent_specifier_types:
        #    if t.name == "requires":
        #        pass

        if specifier_types:
            specifier_node = list_to_typed_node(specifier_types)
            specifier = " " + codegen_type(name_node, specifier_node, cx) + " "

            if any(t.name == "static" for t in specifier_types):
                if const_or_mut:
                    raise CodeGenError("const/mut makes no sense for a static function", const_or_mut[0])
                is_const = False

            def is_template_test(expr):
                return isinstance(expr, Template) and expr.func.name == "template"

            if list(find_all(specifier_node, test=is_template_test)):
                if len(typenames) > 0:
                    raise CodeGenError("Explicit template function with generic params", specifier_node)
                template = ""
            # inline = ""  # debatable whether a non-trailing return should inmply no "inline":
            # no: tvped func above a certain complexity threshold automatically placed in synthesized implementation file

    if interface:
        assert len(typenames) == 0
        template = ""
        inline = ""

    if is_declaration:
        inline = ""

    if typenames:
        template = "template <{0}>\n".format(", ".join(typenames))
        inline = ""

    if name == "main":
        if return_type_node or name_node.declared_type:
            raise CodeGenError("main implicitly returns int. no explicit return type or directives allowed.", defnode)
        template = ""
        inline = ""
        if not isinstance(defnode.parent, Module):
            raise CodeGenError("unexpected nested main function", defnode)
        defnode.parent.has_main_function = True

    if return_type_node:
        # return_type = codegen_type(name_node, name_node.declared_type)
        return_type = codegen_type(defnode, return_type_node, cx)
        if is_destructor:
            raise CodeGenError("destruct methods can't specifiy a return type")
    elif name == "main":
        return_type = "int"
    else:
        return_type = "auto"
        found_return = False
        if block:
            for b in block.args:
                for ret in find_all(b, test=is_return, stop=creates_new_variable_scope):
                    found_return = True
                    if is_void_return(ret):
                        # like python treat 'return' as 'return None' (we change the return type of the defined func to allow deduction of type of '{}' by c++ compiler)
                        # return_type = 'std::shared_ptr<object>'
                        # ^ this is buggy or at least confusing for all sorts of reasons e.g. classes managed by unique_ptr (we're now embracing void)
                        return_type = "void"
                        break

        if not found_return:
            # return_type = 'std::shared_ptr<object>'
            return_type = "void"

    if is_destructor:
        # not marked virtual because inheritance not implemented yet (note that interface abcs have a virtual destructor)
        funcdef = specifier + "~" + class_name + "()" + override + final
    else:
        const = " const" if is_const else ""

        funcdef = "{}{}{}auto {}({}){} -> {}{}{}".format(template, specifier, inline, name, ", ".join(params), const, return_type, override, final)

    indt = cx.indent_str()

    if is_declaration:
        if typenames:
            raise CodeGenError("no declarations with untyped/generic params", defnode)

        if is_method and isinstance(defnode.parent, Assign):

            rhs = defnode.parent.rhs

            if return_type_node is None and not is_destructor:
                raise CodeGenError("declarations must specify a return type", defnode)

            if isinstance(rhs, IntegerLiteral) and rhs.integer_string == "0":

                classdef = cx.lookup_class(class_identifier)
                if not classdef:
                    raise CodeGenError("expected a class/struct in '= 0' declaration", class_identifier)

                classdef.is_pure_virtual = True

                # pure virtual function (codegen_assign handles the "= 0" part)
                return indt + funcdef
            elif rhs.name in ["default", "delete"]:
                return indt + funcdef
            else:
                raise CodeGenError("bad assignment to function declaration", defnode)
        else:
            return indt + funcdef + ";\n\n"

    block_str = codegen_function_body(defnode, block, cx)
    return indt + funcdef + " {\n" + block_str + indt + "}\n\n"


# Replace self.x = y in a method (but not an inner lambda unless by-ref this-capturing) with this->x = y
def _replace_self_with_this(defnode):
    assert defnode.func.name == "def" or isinstance(defnode.func, ArrayAccess) and defnode.func.func.name == "lambda" and any(c.name == "ref" for c in defnode.func.args)

    need_self = False
    for s in find_all(defnode, test=lambda a: a.name == "self", stop=lambda n: n.func and n.func.name in ["class", "struct"]):

        replace_self = True
        p = s
        while p is not defnode:
            if creates_new_variable_scope(p):
                # 'self.foo' inside e.g. a lambda body

                if isinstance(p.func, ArrayAccess) and p.func.func.name == "lambda" and p.func.args and p.func.args[0].name == "ref":
                    # TODO this should also check for &this capture etc
                    pass
                else:
                    replace_self = False
                    break
            p = p.parent

        if replace_self and isinstance(s.parent,
                                       AttributeAccess) and s.parent.lhs is s:
            # rewrite as this->foo:
            this = Identifier("this")
            arrow = ArrowOp("->", [this, s.parent.rhs])
            arrow.scope = s.scope
            this.scope = s.scope
            arrow.parent = s.parent.parent
            this.parent = arrow
            arrow.rhs.parent = arrow

            if s.parent in s.parent.parent.args:
                index = s.parent.parent.args.index(s.parent)
                new_args = s.parent.parent.args
                new_args[index] = arrow
                s.parent.parent.args = new_args  # resetting args (unnecessary in pure python but necessary with pybind11 copying semantics _without_ MAKE_OPAQUE)
            elif s.parent is s.parent.parent.func:
                s.parent.parent.func = arrow
        else:
            need_self = True

    return need_self


def codegen_function_body(defnode : Call, block, cx):
    # methods or functions only (not lambdas!)
    assert defnode.func.name == "def"

    need_self = _replace_self_with_this(defnode)

    block_cx = cx.enter_scope()
    block_cx.in_function_body = True
    block_str = codegen_block(block, block_cx)
    if need_self:
        # (note: this will be a compile error in a non-member function / non-shared_from_this deriving class)
        block_str = block_cx.indent_str() + "const auto self = ceto::shared_from(this);\n" + block_str
    # if not is_destructor and not is_return(block.args[-1]):
    #     block_str += block_cx.indent_str() + "return {};\n"
    # no longer doing this^
    return block_str


def codegen_lambda(node, cx):
    args = list(node.args)
    block = args.pop()
    assert isinstance(block, Block)
    # params = ["auto " + codegen_node(a) for a in args]
    params = []
    for a in args:
        if not isinstance(a, Identifier):
            if isinstance(a, Assign):
                raise CodeGenError("lambda args may not have default values (not supported in C++)", a)
            if isinstance(a, TypeOp):
                # lambda inside decltype on rhs of an outer type case (result of unfortunate choice in sema to not fully flatten TypeOp and only  partially convert to a .declared_type)
                realtype = a.rhs
                a = a.lhs
                a.declared_type = realtype
                if not isinstance(a, Identifier):
                    raise CodeGenError("Unexpected lambda argument for lambda found inside decltype", a)
            else:
                raise CodeGenError("Unexpected lambda argument", a)

        if typed_param := codegen_typed_def_param(a, cx):
            params.append(typed_param)
        else:
            params.append("const auto &" + a.name)
    newcx = cx.enter_scope()
    newcx.in_function_body = True
    type_str = ""
    invocation_str = ""
    if node.declared_type is not None:
        if isinstance(node.declared_type, (TypeOp, Call)):
            if isinstance(node.declared_type, Call):
                return_type_list = []
                last_type = node.declared_type
            elif isinstance(node.declared_type, TypeOp):
                return_type_list = type_node_to_list_of_types(node.declared_type)
                last_type = return_type_list.pop()

            if isinstance(last_type, Call):
                while isinstance(last_type.func, Call):#  and not last_type.func.func.name == "decltype":
                    invocation_str += "(" + ", ".join(codegen_node(a, cx) for a in last_type.args) + ")"
                    last_type = last_type.func

                if (isinstance(last_type.func, Identifier) and last_type.func.name != "decltype"):
                    # this is the case e.g. l = lambda(x:int, 0):int(0)
                    # decltype check avoids the case: l2 = lambda (x: int, 0):decltype(0)  # not an immediately invoked lambda
                    # but note that this is immediately invoked: l3 = lambda(x:int, 0):decltype(1)(2)
                    return_type_list.append(last_type.func)
                    invocation_str += "(" + ", ".join(codegen_node(a, cx) for a in last_type.args) + ")"
                elif isinstance(last_type, Call) and last_type.func.name == "decltype":
                    return_type_list.append(last_type)

                assert return_type_list
                declared_type = list_to_typed_node(return_type_list)
                node.declared_type = declared_type
                type_str = " -> " + codegen_type(node, declared_type, cx)

        if not type_str:
            type_str = " -> " + codegen_type(node, node.declared_type, cx)
    # if isinstance(node.func, ArrowOp):
    #     # not workable for precedence reasons
    #     if declared_type is not None:
    #         raise CodeGenError("Multiple return types specified for lambda?", node)
    #     declared_type = node.func.rhs
    #     # simplify AST (still messed with for 'is return type void?' stuff)
    #     node.declared_type = declared_type  # put type in normal position
    #     node.func = node.func.lhs  # func is now 'lambda' keyword

    capture_this_by_ref = False

    if isinstance(node.func, ArrayAccess):
        # explicit capture list

        def codegen_capture_list_item(a):

            def codegen_capture_list_address_op(u : UnOp):
                if isinstance(u, UnOp) and u.op == "&" and isinstance(u.args[0], Identifier) and not u.args[0].declared_type:
                    # codegen would add parenthese to UnOp arg here:
                    return "&" + codegen_node(u.args[0], cx)
                return None

            nonlocal capture_this_by_ref

            if isinstance(a, Assign):
                if ref_capture := codegen_capture_list_address_op(a.lhs):
                    if a.lhs.args[0].name == "this":
                        capture_this_by_ref = True
                    lhs = ref_capture
                elif isinstance(a.lhs, Identifier) and not a.lhs.declared_type:
                    lhs = codegen_node(a.lhs, cx)
                else:
                    raise CodeGenError("Unexpected lhs in capture list assign", a)
                return lhs + " = " + codegen_node(a.rhs, cx)
            else:
                if ref_capture := codegen_capture_list_address_op(a):
                    if a.args[0].name == "this":
                        capture_this_by_ref = True
                    return ref_capture
                if isinstance(a, UnOp) and a.op == "*" and a.args[0].name == "this":
                    return "*" + codegen_node(a.args[0])
                if not isinstance(a, Identifier) or a.declared_type:
                    raise CodeGenError("Unexpected capture list item", a)
                if a.name == "ref":
                    # special case non-type usage of ref
                    capture_this_by_ref = True
                    return "&"
                elif a.name == "val":
                    if a.scope.find_def(a):
                        raise CodeGenError("no generic 'val' capture allowed because a variable named 'val' has been defined.", a)
                    return "="
                return codegen_node(a, cx)

        capture_list = [codegen_capture_list_item(a) for a in node.func.args]

    elif cx.parent.in_function_body:
        def is_capture(n):
            if not isinstance(n, Identifier):
                return False
            elif isinstance(n.parent, (Call, ArrayAccess, BracedCall, Template)) and n is n.parent.func:
                return False
            elif isinstance(n.parent, AttributeAccess) and n is n.parent.rhs:
                return False
            return True

        # find all identifiers but not call funcs etc or anything in a nested class
        idents = find_all(node, test=is_capture, stop=lambda c: isinstance(c.func, Identifier) and c.func.name in ["class", "struct"])

        idents = {i.name: i for i in idents}.values()  # remove duplicates

        possible_captures = []
        for i in idents:
            if i.name == "self":
                possible_captures.append(i.name)
            elif isinstance(i.parent, Call) and i.parent.func.name in ["def", "lambda"]:
                pass  # don't capture a lambda parameter
            elif (d := i.scope.find_def(i)) and isinstance(d, (LocalVariableDefinition, ParameterDefinition)):
                defnode = d.defined_node
                is_capture = True
                while defnode is not None:
                    if defnode is node:
                        # defined in lambda or by lambda params (not a capture)
                        is_capture = False
                        break
                    defnode = defnode.parent
                if is_capture:
                    possible_captures.append(i.name)

        if isinstance(node.parent, Call) and node is node.parent.func:
            # immediately invoked (TODO: nonescaping)
            # capture by const ref: https://stackoverflow.com/questions/3772867/lambda-capture-as-const-reference/32440415#32440415
            # TODO we still have some work with lowering implicit captures to explicit capture lists as an ast->ast pass in semanticanalysis
            # before this can work (scope handling and _decltype_string with explicit capture lists fixes)
            #capture_list = ["&" + i + " = " + "std::as_const(" + i + ")" for i in possible_captures]
            capture_list = ["&"]  # just allow mutable ref capture until we resolve the above
        else:
            # capture only a few things by const value (shared/weak instances, arithithmetic_v, enums):
            capture_list = [i + " = " + "ceto::default_capture(" + i + ")" for i in possible_captures]
    else:
        capture_list = ""

    if capture_this_by_ref:
        _replace_self_with_this(node)

    return ("[" + ", ".join(capture_list) + "](" + ", ".join(params) + ")" + type_str + " {\n" +
            codegen_block(block, newcx) + newcx.indent_str() + "}" + invocation_str)


def is_super_init(call):
    return isinstance(call, Call) and isinstance(call.func,
        AttributeAccess) and call.func.lhs.name == "super" and call.func.rhs.name == "init"


def is_self_field_access(node):
    if isinstance(node, AttributeAccess) and node.lhs.name == "self":
        if not isinstance(node.rhs, Identifier):
            raise CodeGenError("unexpected attribute access", node)
        return True
    return False


def is_comment(node):
    return isinstance(node, ScopeResolution) and node.lhs.name == "ceto" and (
            isinstance(node.rhs, Call) and node.rhs.func.name == "comment")


def class_name_node_from_inline_method(defcallnode : Call):
    assert isinstance(defcallnode, Call)

    parentblock = defcallnode.parent
    if isinstance(parentblock, Assign):  # e.g. = 0 pure virtual function
        parentblock = parentblock.parent
    if isinstance(parentblock, Block) and isinstance(parentblock.parent, Call) and parentblock.parent.func.name in ["class", "struct"]:
        classname = parentblock.parent.args[0]
        if isinstance(classname, Call):
            # inheritance
            classname = classname.func
        assert isinstance(classname, (Identifier, Template))
        return classname
    return None


def codegen_class(node : Call, cx):
    assert isinstance(node, Call)
    name = node.args[0]
    inherits = None

    if isinstance(name, Call):
        if len(name.args) != 1:
            if len(name.args) == 0:
                raise CodeGenError("empty inherits list", name)
            raise CodeGenError("Multiple inheritance is not supported yet", name)
        inherits = name.args[0]
        name = name.func

    if isinstance(name, Template):
        template_args = name.args
        name = name.func
    else:
        template_args = None

    if not isinstance(name, Identifier):
        raise CodeGenError("bad class first arg", name)

    if len(node.args) not in [1, 2]:
        raise CodeGenError("bad number of args to class", node)

    if len(node.args) == 1:
        is_forward_declaration = True
    else:
        is_forward_declaration = False
        block = node.args[-1]
        if not isinstance(block, Block):
            raise CodeGenError("class missing block (bad second class arg)", node)

    is_template = template_args is not None

    classdef = ClassDefinition(name, node, #is_generic_param_index={},
                               is_unique=node.declared_type and node.declared_type.name == "unique",
                               is_struct=node.func.name == "struct",
                               is_forward_declaration=is_forward_declaration)
    classdef.is_concrete = not is_template

    cx.add_class_definition(classdef)

    defined_interfaces = defaultdict(list)
    local_interfaces = set()
    typenames = []

    indt = cx.indent_str()

    if is_forward_declaration:
        return indt + "struct " + name.name + ";\n\n"

    inner_cx = cx.enter_scope()
    inner_cx.in_class_body = True
    inner_cx.in_function_body = False

    cpp = indt
    inner_indt = inner_cx.indent_str()
    uninitialized_attributes = []
    uninitialized_attribute_declarations : typing.List[str] = []
    constructor_node = None
    constructor_initialized_field_names : typing.List[str] = []
    should_disable_default_constructor = False
    field_types : typing.Dict[str, typing.Union[str, Node]] = {}

    for block_index, b in enumerate(block.args):
        if isinstance(b, Call) and b.func.name == "def":
            methodname = b.args[0]

            if (method_type := b.args[0].declared_type) is not None:
                if isinstance(method_type, Call) and method_type.func.name == "interface" and len(method_type.args) == 1:
                    interface_type = method_type.args[0]
                    if methodname.name in ["init", "destruct"]:
                        raise CodeGenError("init or destruct can't defined as interface methods", b)

                    #if interface_type.name in defined_interfaces or not any(t == interface_type.name for t in cx.interfaces):
                    if interface_type.name in defined_interfaces or not isinstance(cx.lookup_class(interface_type), InterfaceDefinition):
                        defined_interfaces[interface_type.name].append(b)

                    cx.add_interface_method(interface_type.name, b)
                    local_interfaces.add(interface_type.name)

            if methodname.name == "init":
                if constructor_node is not None:
                    raise CodeGenError("Delegating constructors are not yet implemented (only one init method allowed for now)", b)
                constructor_node = b
            else:
                funcx = inner_cx.enter_scope()
                funcx.in_function_param_list = True
                cpp += codegen_def(b, funcx)
        elif isinstance(b, Identifier):

            if b.name == "pass":
                continue
            elif b.declared_type is None:
                # generic case

                if template_args is not None:
                    raise CodeGenError("no generic params with an explicit template class")

                t = gensym("C")
                typenames.append(t)
                field_types[b.name] = t
                decl_const_part = ""
                decl = decl_const_part + t + " " + b.name
                cpp += inner_indt + decl + ";\n\n"
                # classdef.is_generic_param_index[block_index] = True
                is_template = True
            else:
                field_type = b.declared_type
                decl = codegen_type(b, b.declared_type, inner_cx) + " " + b.name
                field_types[b.name] = field_type
                cpp += inner_indt + decl + ";\n\n"
                # classdef.is_generic_param_index[block_index] = False

            uninitialized_attributes.append(b)
            uninitialized_attribute_declarations.append(decl)
        elif isinstance(b, Assign):
            cpp += inner_indt + codegen_assign(b, inner_cx) + ";\n\n"
        elif is_comment(b):
            cpp += codegen_node(b, inner_cx)
        else:
            raise CodeGenError("Unexpected expression in class body", b)

    # classdef.is_concrete = not classdef.is_generic_param_index
    classdef.is_concrete = not is_template

    base_class_type : typing.Optional[str] = None
    if isinstance(inherits, Identifier):
        # TODO this avoids some error checking (many uses of 'name' have the same issue)
        base_class_type = inherits.name
        inherits_dfn = cx.lookup_class(inherits)
        if inherits_dfn and not inherits_dfn.is_concrete:
            classdef.is_concrete = False
    if isinstance(inherits, (AttributeAccess, ScopeResolution)):
        # when ceto defined namespaces implemented this will have to evaluate
        # inherits without adding shared_ptr etc like in the Identifier case
        base_class_type = codegen_node(inherits, cx)
    elif isinstance(inherits, Template):
        base_class_type = inherits.func.name + "<" + ", ".join(codegen_node(t, cx) for t in inherits.args) + ">"

    fields_used_only_in_initializer_list = set()
    args_only_passed_to_super_init = set()

    if constructor_node is not None:
        constructor_args = constructor_node.args[1:-1]
        constructor_block = constructor_node.args[-1]
        assert isinstance(constructor_block, Block)
        initializerlist_assignments = []
        super_init_call = None
        params_initializing_fields = []
        initcx = inner_cx.enter_scope()
        # initcx.in_function_param_list = True  # TODO remove this field

        for stmt in constructor_block.args:

            # handle self.whatever = something
            if isinstance(stmt, Assign) and is_self_field_access(stmt.lhs):
                field = stmt.lhs.rhs
                initializerlist_assignments.append((field, stmt.rhs))
                constructor_initialized_field_names.append(field.name)
                params_initializing_fields.extend(
                    (a, field.name) for a in constructor_args if
                    a.name == stmt.rhs.name)
                if isinstance(stmt.rhs, Identifier) and is_last_use_of_identifier(stmt.rhs):
                    fields_used_only_in_initializer_list.add(field.name)
            elif is_super_init(stmt):
                if super_init_call is not None:
                    raise CodeGenError("only one call to super.init allowed", stmt)
                super_init_call = stmt
                for a in super_init_call.args:
                    if isinstance(a, Identifier) and is_last_use_of_identifier(a):
                        args_only_passed_to_super_init.add(a.name)
            else:
                # anything that follows won't be printed as an initializer-list assignment/base-class-constructor-call
                break

        # constructor_block.args = constructor_block.args[
        #                          len(initializerlist_assignments) + int(super_init_call is not None):]
        constructor_block_new = Block(constructor_block.args[len(initializerlist_assignments) + int(super_init_call is not None):])
        # TODO ^ this doesn't set .parent. Otoh all of its args' parents point to the old block. Might be ok in practice or might be the cause of future/uncovered bugs (find_defs walks .parent)

        if any(find_all(constructor_block_new, is_super_init)):
            raise CodeGenError("A call to super.init must occur in the 'initializer list' (that is, any statements before super.init must be of the form self.field = val")

        init_params = []
        init_param_type_from_name = {}
        should_std_move_constructor_param_names = set()

        for arg in constructor_args:
            if not isinstance(arg, Assign):
                should_disable_default_constructor = True

            if typed_arg_tuple := _codegen_typed_def_param_as_tuple(arg, initcx):
                typed_arg_str_lhs, typed_arg_str_rhs = typed_arg_tuple
                init_params.append(typed_arg_str_lhs + " " + typed_arg_str_rhs)
                if isinstance(arg, (Assign, NamedParameter)):
                    assert arg.lhs.name
                    init_param_type_from_name[arg.lhs.name] = typed_arg_str_lhs
                else:
                    assert isinstance(arg, Identifier)
                    init_param_type_from_name[arg.name] = typed_arg_str_lhs

            elif isinstance(arg, Identifier):
                # generic constructor arg:

                found_type = False
                for param, field_name in params_initializing_fields:
                    if arg is param and field_name in field_types:
                        field_type = field_types[field_name]
                        found_type = True
                        if isinstance(field_type, Node):
                            # mutate the ast so that we print arg with proper "lists/strings/objects const ref in func params" behavior
                            # TODO this is not the worst thing (which would be byval + a copy) but we should be std::move-ing if self.a = a is the last use of param a (at least in the same circumstances as we do for uninitialized variables constructor synthesis)
                            arg.declared_type = field_type
                            typed_arg_str_lhs, typed_arg_str_rhs = _codegen_typed_def_param_as_tuple(arg, initcx)  # this might add const& etc
                            init_params.append(typed_arg_str_lhs + " " + typed_arg_str_rhs)
                            init_param_type_from_name[arg.name] = typed_arg_str_lhs
                        else:
                            # generic field case
                            init_param = field_type + " " + arg.name
                            if field_name in fields_used_only_in_initializer_list:
                                should_std_move_constructor_param_names.add(arg.name)
                            else:
                                init_param = "const " + init_param
                            init_params.append(init_param)
                            init_param_type_from_name[arg.name] = field_type

                if not found_type:
                    if template_args:
                        raise CodeGenError("no generic params in constructor for explicit template class", node)

                    t = gensym("C")
                    typenames.append(t)
                    init_param_type_from_name[arg.name] = t
                    if arg.name in args_only_passed_to_super_init:
                        init_params.append(t + " " + arg.name)
                    else:
                        init_params.append("const " + t + "& " + arg.name)
            else:
                raise CodeGenError("unexpected constructor arg", b)

        cpp += inner_indt + "explicit " + name.name + "(" + ", ".join(init_params) + ")"

        def initializer_list_initializer_from_field_param(field, param):
            field_code = codegen_node(field, initcx)
            rhs_code = codegen_node(param, initcx)
            if param.name in should_std_move_constructor_param_names and constructor_node is None:
                rhs_code = "std::move(" + rhs_code + ")"
            return field_code + "(" + rhs_code + ")"

        initializer_list_items = [
            initializer_list_initializer_from_field_param(field, rhs)
            for field, rhs in initializerlist_assignments]

        if super_init_call is not None:
            if len(super_init_call.args) == 0:
                raise CodeGenError("no explicit calls to super.init() without args (just let c++ do this implicitly)")

            super_init_fake_args = []

            for arg in super_init_call.args:
                if isinstance(arg, Identifier) and arg.name in init_param_type_from_name:
                    # forward the type of the constructor arg to the base class constructor call
                    super_init_fake_args.append("std::declval<" + init_param_type_from_name[arg.name] +  ">()")
                elif is_self_field_access(arg):  # this would fail in C++
                    raise CodeGenError("no reads from self in super.init call", arg)
                else:
                    super_init_fake_args.append(codegen_node(arg, inner_cx))

            super_init_args = []
            for a in super_init_call.args:
                arg_code = codegen_node(a, inner_cx)
                if a.name in args_only_passed_to_super_init and constructor_node is None:
                    arg_code = "std::move(" + arg_code + ")"
                super_init_args.append(arg_code)

            inherits_dfn = cx.lookup_class(inherits)
            if inherits_dfn and not inherits_dfn.is_concrete:
                classdef.is_concrete = False

            if inherits_dfn and not inherits_dfn.is_concrete and not inherits_dfn.is_pure_virtual and isinstance(inherits, Identifier):
                # TODO lookup_class should maybe ignore forward_declarations when the full definition is available. Alt
                # here CTAD takes care of the real type of the base class (in case the base class is a template)
                # see https://stackoverflow.com/questions/74998572/calling-base-class-constructor-using-decltype-to-get-more-out-of-ctad-works-in
                base_class_type = "decltype(" + inherits.name + "(" + ", ".join(super_init_fake_args) + "))"
                # this is only necessary for msvc (see above link) but it's fine on other platforms:
                base_class_type = "std::type_identity_t<" + base_class_type + ">"

            super_init_str = base_class_type + " (" + ", ".join(super_init_args) + ")"
            initializer_list_items.insert(0, super_init_str)  # TODO we should preserve the user defined order

        initializer_list = ", ".join(initializer_list_items)

        if initializer_list:
            cpp += " : " + initializer_list + " "

        cpp += "{\n"
        cpp += codegen_function_body(constructor_node, constructor_block_new, initcx)
        cpp += inner_indt + "}\n\n"

    uninitialized_attributes = [u for u in uninitialized_attributes if u.name not in constructor_initialized_field_names]

    if uninitialized_attributes:
        if constructor_node is not None:
            raise CodeGenError("class {} defines a constructor (init method) but does not initialize these attributes: {}".format(name.name, ", ".join(str(u) for u in uninitialized_attributes)))

        # TODO this should also be used in the case of a typed constructor param
        def should_std_move_constructor_param(a):
            if not a.declared_type or (
                    _should_add_const_ref_to_typed_param(a, cx) or (
                    cd := _class_def_from_typed_param(a, cx)) and cd.is_unique):
                return True
            return False

        def initializer_list_initializer(a):
            if should_std_move_constructor_param(a) and constructor_node is None:
                return a.name + "(std::move(" + a.name + "))"
            else:
                return a.name + "(" + a.name + ")"

        # autosynthesize constructor
        cpp += inner_indt + "explicit " + name.name + "(" + ", ".join(uninitialized_attribute_declarations) + ") : "
        cpp += ", ".join([initializer_list_initializer(a) for a in uninitialized_attributes]) + " {}\n\n"
        should_disable_default_constructor = True

    if should_disable_default_constructor:
        # this matches python behavior (though we allow a default constructor for a class with no uninitialized attributes and no user defined constructor)
        cpp += inner_indt + name.name + "() = delete;\n\n"

    interface_def_str = ""
    for interface_type in defined_interfaces:
        interface_def_str += "struct " + interface_type + " : ceto::object {\n"  # no longer necessary to inherit from object to participate in autoderef but we're keeping it for now (would be required if we re-enabled call_or_construct)
        for method in defined_interfaces[interface_type]:
            print("method",method)
            interface_def_str += inner_indt + interface_method_declaration_str(method, cx)
        interface_def_str += inner_indt + "virtual ~" + interface_type + "() = default;\n\n"
        interface_def_str +=  "};\n\n"

    cpp += indt + "};\n\n"

    default_inherits = ["public " + i for i in local_interfaces]
    if base_class_type is not None:
        default_inherits.append("public " + base_class_type)

    if not inherits:
        # TODO non-class inheritance should still require deriving from ceto::[shared_]object
        if classdef.is_unique or classdef.is_struct:
            # no longer necessary for autoderef (but would be required for call_or_construct)
            default_inherits += ["public ceto::object"]
        else:
            # maybe TODO stop doing this automatically (non-trivial self use ie ceto::shared_from will fail to compile without the explicit opt-in)
            # but regardless TODO need multiple-inheritance as long as 2nd+ base is not a ceto (shared) class

            if is_template:
                # For classes whose type depends on a single constructor call (via ctad) we could do better
                # e.g std::enable_shared_from_this<decltype(Foo{std::declval<int>(), ...})>
                default_inherits += ["public ceto::enable_shared_from_this_base_for_templates"]
            else:
                default_inherits += ["public ceto::shared_object", "public std::enable_shared_from_this<" + name.name + ">"]

    #class_header = 'struct __attribute__((visibility("default")))' + name.name + " : " + ", ".join(default_inherits)
    class_header = "struct " + name.name + " : " + ", ".join(default_inherits)
    class_header += " {\n\n"

    if inherits and not (constructor_node or uninitialized_attributes):
        # There may be gotchas inheriting the base class constructors automatically especially for
        # external C++ defined classes (also will need fixes for multiple inheritance!).
        # Though can be avoided by defining an explicit init method.

        if isinstance(inherits, Identifier):
            class_header += "using " + base_class_type + "::" + base_class_type + ";\n\n"
        elif isinstance(inherits, (ScopeResolution, AttributeAccess)):
            flattened = nested_same_binop_to_list(inherits)
            if isinstance(flattened[-1], (ScopeResolution, AttributeAccess)):
                raise CodeGenError("mix of ScopeResolution and AttributeAccess in inherits list - pick one or the other", inherits)
            if isinstance(flattened[-1], Identifier):
                class_header += "using " + base_class_type + "::" + codegen_node(flattened[-1], cx) + ";\n\n"

    if typenames:
        template_header = "template <" + ", ".join(["typename " + t for t in typenames]) + ">"
    elif template_args:
        template_header = "template <" + ", ".join(codegen_node(t, cx) for t in template_args) + ">"
    else:
        template_header = ""

    return interface_def_str + template_header + class_header + cpp


def interface_method_declaration_str(defnode: Call, cx):
    assert defnode.func.name == "def"
    name_node = defnode.args[0]
    name = name_node.name
    args = defnode.args[1:]
    block = args.pop()
    assert isinstance(block, Block)

    params = []
    return_type_node = defnode.declared_type

    if return_type_node is None:
        raise CodeGenError("must specify return type of interface method")
    return_type = codegen_type(defnode, return_type_node, cx)

    is_const = not mut_by_default

    specifier_node = name_node.declared_type

    type_nodes = type_node_to_list_of_types(specifier_node)

    const_or_mut = [t for t in type_nodes if t.name in ["const", "mut"]]
    if len(const_or_mut) > 1:
        raise CodeGenError("too many 'mut' and 'const' specified", defnode)

    if const_or_mut:
        if const_or_mut[0].name == "const":
            is_const = True
        elif const_or_mut[0].name == "mut":
            is_const = False

        type_nodes.remove(const_or_mut[0])

    const = " const" if is_const else ""
    specifier = ""

    type_nodes = [t for t in type_nodes if not (isinstance(t, Call) and t.func.name == "interface")]

    if type_nodes:
        specifier_node = list_to_typed_node(type_nodes)
        specifier = " " + codegen_type(name_node, specifier_node, cx) + " "

    for i, arg in enumerate(args):
        if arg.declared_type is None:
            raise CodeGenError("parameter types must be specified for interface methods")
        if not isinstance(arg, Identifier):
            raise CodeGenError("Only simple args allowed for interface method (you don't want c++ virtual functions with default arguments)")
        param = codegen_typed_def_param(arg, cx)
        assert len(param) > 0
        params.append(param)

    return "{}virtual {} {}({}){} = 0;\n\n".format(specifier, return_type, name, ", ".join(params), const)


def codegen_while(whilecall, cx):
    assert isinstance(whilecall, Call)
    assert whilecall.func.name == "while"
    if len(whilecall.args) != 2:
        raise CodeGenError("Incorrect number of while args", whilecall)
    if not isinstance(whilecall.args[1], Block):
        raise CodeGenError("Last while arg must be a block", whilecall.args[1])

    cpp = "while (" + codegen_node(whilecall.args[0], cx.enter_scope()) + ") {"
    cpp += codegen_block(whilecall.args[1], cx.enter_scope())
    cpp += cx.indent_str() + "}\n"
    return cpp


def codegen_if(ifcall : Call, cx):
    assert isinstance(ifcall, Call)
    assert ifcall.func.name == "if"

    indt = cx.indent_str()
    cpp = ""

    is_expression = not isinstance(ifcall.parent, Block)

    if is_expression:
        if any(find_all(ifcall, is_return, stop=creates_new_variable_scope)):
            raise CodeGenError("no explicit return in if expression", ifcall)

        for a in ifcall.args:
            if isinstance(a, Block):
                last_statement = a.args[-1]
                if not (isinstance(last_statement, Call) and last_statement.func.name == "throw"):
                    synthetic_return = SyntaxTypeOp(":", [Identifier("return"), last_statement])
                    last_statement.parent = synthetic_return
                    synthetic_return.parent = a
                    a.args = a.args[0:-1] + [synthetic_return]

    ifnode = IfWrapper(ifcall.func, ifcall.args)

    if ifcall.declared_type:
        ifkind = ifcall.declared_type.name
        if ifkind is None:
            raise CodeGenError("unexpected if type", ifcall)
    else:
        ifkind = None

    if_start = "if ("
    block_opening = ") {\n"
    elif_start = "} else if ("
    else_start = "} else {\n"
    block_closing = "}"

    if ifkind == "noscope":
        # python-style "noscope" ifs requires a specifier (TODO probably should just remove this silliness)

        if is_expression or not cx.in_function_body:
            raise CodeGenError("unscoped if disallowed in expression context", ifcall)

        # scopes = [ifnode.cond, ifnode.thenblock]
        scopes = [ifnode.thenblock]
        if ifnode.elseblock is not None:
            scopes.append(ifnode.elseblock)
        else:
            raise CodeGenError("unscoped if without else block disallowed", ifcall)

        for elifcond, elifblock in ifnode.eliftuples:
            # scopes.append(elifcond)
            scopes.append(elifblock)

        assigns = { scope:[] for scope in scopes }

        for scope in scopes:
            assigns[scope].extend(find_all(scope, test=lambda n: isinstance(n, Assign), stop=creates_new_variable_scope))

        print("all if assigns", list(assigns))

        declarations = { scope:{} for scope in scopes }

        for scope in scopes:
            for assign in assigns[scope]:
                if hasattr(assign, "already_declared"):
                    continue
                if isinstance(assign.lhs, Identifier) and not assign.lhs.declared_type and not assign.scope.find_def(assign.lhs):
                    if assign.lhs.name in declarations[scope]:
                        continue
                    declarations[scope][assign.lhs.name] = assign, codegen_node(assign.rhs, cx)
                else:
                    raise CodeGenError("unexpected non-simple assign in noscope if", assign)

        if all(declarations[scopes[0]].keys() == declarations[s].keys() for s in scopes[1:]):
            # allow noscope

            for lhs in declarations[scopes[0]]:
                assign, declname = declarations[scopes[0]][lhs]
                assign.already_declared = True
                cpp += f"decltype({declname}) {lhs};\n" + indt
            for scope in scopes[1:]:
                for lhs in declarations[scope]:
                    assign, declname = declarations[scope][lhs]
                    assign.already_declared = True
        else:
            raise CodeGenError("unbalanced assignments in if prevents noscope", ifnode)

    # TODO should these be disallowed with expression ifs?
    elif ifkind == "preprocessor":
        if_start = "#if "
        block_opening = "\n"
        elif_start = "#elif "
        else_start = "#else\n"
        block_closing = "#endif\n"
    elif ifkind == "constexpr":
        if_start = "if constexpr ("
    elif ifkind == "consteval":  # c++23
        if_start = "if consteval ("

    if isinstance(ifnode.cond, NamedParameter) and not ifcall.is_one_liner_if:
        raise CodeGenError("assignment in if missing extra parenthesese", ifnode.cond)

    cpp += if_start + codegen_node(ifnode.cond, cx) + block_opening

    cpp += codegen_block(ifnode.thenblock, cx.enter_scope())

    for elifcond, elifblock in ifnode.eliftuples:
        cpp += indt + elif_start + codegen_node(elifcond, cx.enter_scope()) + block_opening
        cpp += codegen_block(elifblock, cx.enter_scope())

    if ifnode.elseblock:
        cpp += indt + else_start
        cpp += codegen_block(ifnode.elseblock, cx.enter_scope())

    cpp += indt + block_closing

    if is_expression:
        if ifkind is not None:
            raise CodeGenError(f"An expression-if can't be used with a 'type', namely {ifkind}", ifnode)
        if cx.in_function_body:
            capture = "&"
        else:
            capture = ""
        cpp = "[" + capture + "]() {" + cpp + "}()"

    return cpp + "\n"


def codegen_for(node, cx):

    assert isinstance(node, Call)
    if len(node.args) != 2:
        raise CodeGenError("'for' must have two arguments - the iteration part and the indented block. 'One liner' for loops are not supported.", node)
    arg, block = node.args
    if not isinstance(block, Block):
        raise CodeGenError("expected block as last arg of for", node)

    iter_type : Node = None

    instmt = node.args[0]

    if not isinstance(instmt, BinOp) or instmt.op != "in":
        raise CodeGenError("unexpected 1st argument to for", node)

    var = instmt.lhs
    if not isinstance(var, (Identifier, TupleLiteral)):
        raise CodeGenError("Unexpected iter var", var)

    iterable = instmt.rhs

    if isinstance(var, TupleLiteral):
        structured_binding = _structured_binding_unpack_from_tuple(var, True, cx)
        type_str = " "
        var_str = structured_binding
    elif var.declared_type is not None:
        assert isinstance(var, Identifier)
        type_str, var_str = _codegen_typed_def_param_as_tuple(var, cx)
    else:
        var_str = codegen_node(var, cx)
        type_str = None

    if type_str is None:
        type_str = "const auto&"

    iterable_str = codegen_node(iterable, cx)
    rng = gensym("rng")
    idx = gensym("idx")
    initial_list_size = gensym("size")

    indt = cx.indent_str()
    block_cx = cx.enter_scope()

    has_return = any(isinstance(a, SyntaxTypeOp) and a.lhs.name == "return" for a in block.args)

    block_str = codegen_block(block, block_cx)

    # https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2020/p2012r0.pdf
    # use a lambda with a param to extend lifetime 
    # (apparently fixed for actual range based for in c++23 https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2718r0.html)
    # Note: exception handling not ideal for return - though avoids issues supporting returning a reference (even if if only allowed in unsafe mode)
    # - should break apart iter to extend lifetimes without the lambda - but this is fine for now
    # Also breaks goto (ok) and needs "yield" fixes

    if has_return:
        return_type = " -> decltype(auto) "
        throw_stmt = "throw ceto::EndLoopMarkerError();"
    else:
        return_type = ""
        throw_stmt = ""

    forstr = rf"""[&](auto&& {rng}){return_type}{{
    size_t {initial_list_size} = std::size({rng});
    for (size_t {idx} = 0; ; {idx}++) {{
        if (std::size({rng}) != {initial_list_size}) {{
            std::cerr << "Container size changed during iteration: " << __FILE__ << " line: "<< __LINE__ << "\n";
            std::terminate();
        }}
        if ({idx} >= {initial_list_size}) {{
            break;
        }}
        {type_str} {var_str} = {rng}[{idx}];
        {block_str}
    }}
    {throw_stmt}
}}({iterable_str});
    """

    if has_return:
        forstr = rf"""try {{
    return {forstr};
}} catch (const ceto::EndLoopMarkerError&) {{
    ; // pass (loop end)
}}"""

    return textwrap.indent(forstr, indt)


def codegen_range_based_for(node, cx):
    assert isinstance(node, Call)
    if len(node.args) != 2:
        raise CodeGenError("'for' must have two arguments - the iteration part and the indented block. 'One liner' for loops are not supported.", node)
    arg, block = node.args
    if not isinstance(block, Block):
        raise CodeGenError("expected block as last arg of for", node)

    iter_type : Node = None

    instmt = node.args[0]

    if not isinstance(instmt, BinOp) or instmt.op != "in":
        raise CodeGenError("unexpected 1st argument to for", node)

    var = instmt.lhs
    if not isinstance(var, (Identifier, TupleLiteral)):
        raise CodeGenError("Unexpected iter var", var)

    iterable = instmt.rhs

    if isinstance(var, TupleLiteral):
        structured_binding = _structured_binding_unpack_from_tuple(var, True, cx)
        type_str = " "
        var_str = structured_binding
    elif var.declared_type is not None:
        assert isinstance(var, Identifier)
        type_str, var_str = _codegen_typed_def_param_as_tuple(var, cx)
    else:
        var_str = codegen_node(var, cx)
        type_str = None

    if type_str is None:
        type_str = "const auto&"

    forstr = 'for({} {} : {}) {{\n'.format(type_str, var_str, codegen_node(iterable, cx))

    indt = cx.indent_str()
    block_cx = cx.enter_scope()
    forstr += codegen_block(block, block_cx)
    forstr += indt + "}\n"
    return forstr


def _is_unique_var(node: Identifier, cx: Scope):
    # should be handled in a prior typechecking pass (like most uses of find_defs)

    def _is(defining):
        declared_type = strip_mut_or_const(defining.declared_type)
        classdef = cx.lookup_class(declared_type)
        if classdef and classdef.is_unique:
            return True
        elif isinstance(defining, Assign) and isinstance(rhs := strip_mut_or_const(defining.rhs), Call) \
                and (classdef := cx.lookup_class(rhs.func)) and classdef.is_unique:
            return True

    assert isinstance(node, Identifier)

    if not node.scope:
        # nodes on rhs of TypeOp currently don't have a scope
        return False

    for defn in node.scope.find_defs(node):
        if isinstance(defn, (LocalVariableDefinition, ParameterDefinition)):
            if _is(defn.defining_node):
                return True
    if isinstance(node.parent, Assign) and _is(node.parent):
        return True

    return False


def is_last_use_of_identifier(node: Identifier):
    assert isinstance(node, Identifier)
    name = node.name
    if name == "u" and node in node.parent.args:
        pass
    ident_ancestor = node.parent
    prev_ancestor = node
    is_last_use = True
    if isinstance(node.parent,
                  (ScopeResolution, AttributeAccess, ArrowOp, Template)):
        is_last_use = False
    elif isinstance(node.parent, (BinOp, UnOp)):
        is_last_use = isinstance(node.parent,
                                 Assign)  # maybe a bit strict but e.g. we don't want to transform &x to &(std::move(x))
    elif isinstance(node.parent, Call) and node.parent.func is node:
        is_last_use = False

    # apparently unnecessary
    #elif isinstance(node.parent, Call) and node in node.parent.args:
    #    # if node appears in Call args more than once it's not a last use (regardless of order of evaluation of the Call args)
    #    for arg in [n for n in node.parent.args if n is not node]:
    #        if any(find_all(arg, lambda n: n is not node and n.name == node.name)):
    #            is_last_use = False

    while is_last_use and ident_ancestor and not creates_new_variable_scope(
            ident_ancestor):
        if isinstance(ident_ancestor,
                      Block) and prev_ancestor in ident_ancestor.args:
            # TODO 'find_uses' in sema should work with Identifier not just Assign too
            for b in ident_ancestor.args[ident_ancestor.args.index(prev_ancestor) + 1:]:
                if any(find_all(b, lambda n: (n is not node) and (
                        n.name == name))):
                    is_last_use = False
                    break
        if any(n and n is not node and isinstance(n, Node) and name == n.name
               #for n in ident_ancestor.args + [ident_ancestor.func]):
               for n in ident_ancestor.args):
            is_last_use = False
            break
        prev_ancestor = ident_ancestor
        ident_ancestor = ident_ancestor.parent
    return is_last_use


def codegen_attribute_access(node: AttributeAccess, cx: Scope):
    assert isinstance(node, AttributeAccess)

    if isinstance(node.lhs, Identifier) and cx.lookup_class(node.lhs):  # TODO: bad separate lookup mechanism for class defs and variable defs won't work with shadowing of a class name by a local (legal in C++)
        if node.rhs.name == "class":
            # one might need the raw class name Foo rather than shared_ptr<(const)Foo> without going through hacks like std.type_identity_t<Foo:mut>::element_type.
            # Note that Foo.class.static_member is not handled (resulting in a C++ error for such code) - good because Foo.static_member already works even for externally defined C++ classes
            # TODO ^ needs test
            return node.lhs.name

        # TODO there's a bug/misfeature where class names are registered as VariableDefs (there's a similar bug with function def names that 'does the right thing for the wrong reasons' w.r.t e.g. lambda capture - will eventually need fixing too)
        return node.lhs.name + "::" + codegen_node(node.rhs, cx)

    if isinstance(node.lhs, (Identifier, AttributeAccess)):
        # implicit scope resolution:

        # Here we implement for example: in A.b.c, if A doesn't lookup as a variable, print whole thing as scope resolution A::b::c. Note this makes e.g. accessing data members of globals defined in external C++ headers impossible (good!). TODO Such accesses can be allowed in the future once a python-style 'global' is implemented

        # e.g. std.q.r.s should print as std::q::r::s
        # HOWEVER std.q.r().s should print as std::q::r().s  where the last '.' is a maybe autoderef

        leading = node
        scope_resolution_list = []

        while isinstance(leading, AttributeAccess):
            scope_resolution_list.append(leading.rhs)
            leading = leading.lhs

        if isinstance(leading, Identifier) and leading.name != "self" and not leading.scope.find_def(leading):

            # I think we can get away without overparenthesizing chained scope resolutions (in C++ :: binds tightest so is not actually left associative - https://learn.microsoft.com/en-us/cpp/cpp/cpp-built-in-operators-precedence-and-associativity?view=msvc-170 is a better reference than https://en.cppreference.com/w/cpp/language/operator_precedence here)
            scope_resolution_code = leading.name
            while scope_resolution_list:
                if isinstance(scope_resolution_list[0], Identifier):
                    item = scope_resolution_list.pop()
                    scope_resolution_code += "::" + item.name
                else:
                    break

            if not scope_resolution_list:
                return scope_resolution_code
            else:
                remaining = list_to_attribute_access_node(scope_resolution_list)
                assert remaining is not None
                return scope_resolution_code + "::" + codegen_node(remaining, cx)

    # maybe autoderef
    # we're preferring *(...). instead of -> due to overloading concerns. -> might be fine so long as we're only autoderefing shared/unique/optional

    if node.rhs.name in ["value", "has_value", "value_or", "and_then", "transform", "or_else", "swap", "reset", "emplace"]:
        # don't autoderef an optional if we're calling a method of std::optional on it (maybe this is dubious for the mutable methods swap/reset/emplace?)
        return "(*ceto::mad_smartptr(" + codegen_node(node.lhs, cx) + "))." + codegen_node(node.rhs, cx)

    # autoderef optional or smart pointer:
    return "(*ceto::mad(" + codegen_node(node.lhs, cx) + "))." + codegen_node(node.rhs, cx)


def extract_mut_or_const(type_node : Node) -> typing.Tuple[Node, typing.List[Node]]:
    if type_node is None:
        return None
    types = type_node_to_list_of_types(type_node)
    mc = list(t for t in types if t.name in ["mut", "const"])
    if len(mc) == 1:
        types.remove(mc[0])
        return mc[0], list_to_typed_node(types)
    return None


def strip_mut_or_const(type_node : Node):
    if mc := extract_mut_or_const(type_node):
        return mc[1]
    return type_node


def _class_def_from_typed_param(param, cx):
    assert param.declared_type is not None
    type_node = strip_mut_or_const(param.declared_type)
    # note that mut:Foo (or Foo:mut), that is shared_ptr<Foo>, should still be passed by const ref
    if class_def := cx.lookup_class(type_node):
        return not class_def


def _should_add_const_ref_to_typed_param(param, cx):
    assert param.declared_type is not None
    type_node = strip_mut_or_const(param.declared_type)
    # note that mut:Foo (or Foo:mut), that is shared_ptr<Foo>, should still be passed by const ref
    if class_def := cx.lookup_class(type_node):
        return not class_def.is_unique
    if isinstance(param.declared_type, (ListLiteral, TupleLiteral)):
        return True
    if isinstance(param.declared_type, AttributeAccess) and param.declared_type.rhs.name == "class" and cx.lookup_class(param.declared_type.lhs):
        return True
    if param.declared_type.name == "string" or isinstance(param.declared_type, (AttributeAccess, ScopeResolution)) and param.declared_type.lhs.name == "std" and param.declared_type.rhs.name == "string":
        return True
    return False


def codegen_typed_def_param(arg, cx):  # or default argument e.g. x=1
    t = _codegen_typed_def_param_as_tuple(arg, cx)
    if t:
        return " ".join(t)
    return None


def _strip_non_class_non_plain_mut_type(node : Node, cx):
    assert isinstance(node, Node)
    assert node.declared_type is not None
    types = type_node_to_list_of_types(node.declared_type)

    if len(types) <= 1:
        # don't strip 'mut' alone
        return None

    muts_only = [t for t in types if t.name == "mut"]

    num_muts = len(muts_only)
    if num_muts > 1:
        raise CodeGenError("too many 'mut' specifiers", node)

    if num_muts == 0:
        return None

    mut = muts_only[0]
    mut_index = types.index(mut)

    to_left_of_mut = None
    to_right_of_mut = None

    if mut_index > 0:
        to_left_of_mut = types[mut_index - 1]

    if mut_index < len(types) - 1:
        to_right_of_mut = types[mut_index + 1]

    if any(t for t in [to_left_of_mut, to_right_of_mut] if t and (cdef := cx.lookup_class(t)) and not cdef.is_struct):
        # don't strip 'mut' if adjacent to a class type.
        # TODO this might need adjusting to support Foo:mut:shared or Foo:shared:mut when Foo is of struct type?
        return None

    types.remove(mut)
    return list_to_typed_node(types)


def _ensure_auto_or_ref_specifies_mut_const(type_nodes):
    if ("auto" in type_nodes or "ref" in type_nodes) and not ("const" in type_nodes or "mut" in type_nodes):
        raise CodeGenError("you must specify const/mut", type_nodes[0])


def _codegen_typed_def_param_as_tuple(arg, cx):

    should_add_outer_const = not mut_by_default
    stripped_mut = False

    # TODO should handle plain 'mut'/'const' param (generic case)

    if arg.declared_type is not None:
        if isinstance(arg, Identifier):
            _ensure_auto_or_ref_specifies_mut_const(type_node_to_list_of_types(arg))

            arg_type = arg.declared_type
            if stripped := _strip_non_class_non_plain_mut_type(arg, cx):
                arg_type = stripped
                should_add_outer_const = False
                stripped_mut = True
            elif any(t.name == "const" for t in type_node_to_list_of_types(arg.declared_type)):
                should_add_outer_const = False
                stripped_mut = True

            automatic_const_part = " "  # TODO add const here (if not already const)
            if should_add_outer_const:
                if (class_def := cx.lookup_class(strip_mut_or_const(arg.declared_type))) and class_def.is_unique:
                    # don't add const to unique managed param (it will impede automatic std::move from last use)
                    pass
                else:
                    automatic_const_part = "const "

            automatic_ref_part = ""

            if not stripped_mut and _should_add_const_ref_to_typed_param(arg, cx):
                automatic_const_part = "const "
                automatic_ref_part = "&"

            # treat e.g. external C++ types verbatim (except for adding 'const'
            type_part = automatic_const_part + codegen_type(arg, arg_type, cx) + automatic_ref_part
            return type_part, " " + arg.name
        else:
            # params.append(autoconst(autoref(codegen_type(arg, arg.declared_type, cx))) + " " + codegen_node(arg, cx))
            # note precedence change making
            raise SemanticAnalysisError(
                "unexpected typed expr in defs parameter list", arg)
    elif isinstance(arg, Assign) and not isinstance(arg, NamedParameter):
        raise SemanticAnalysisError(
            "Overparenthesized assignments in def parameter lists are not treated as named params. To fix, remove the redundant parenthesese from:",
            arg)
    elif isinstance(arg, NamedParameter):
        if not isinstance(arg.lhs, Identifier):
            raise SemanticAnalysisError(
                "Non identifier left hand side in def arg", arg)

        if arg.lhs.declared_type is not None:
            _ensure_auto_or_ref_specifies_mut_const(type_node_to_list_of_types(arg.lhs.declared_type))

            arg_type = arg.lhs.declared_type
            if stripped := _strip_non_class_non_plain_mut_type(arg.lhs, cx):
                arg_type = stripped
                should_add_outer_const = False
            elif any(t.name == "const" for t in type_node_to_list_of_types(arg.lhs.declared_type)):
                should_add_outer_const = False

            automatic_const_part = " "
            automatic_ref_part = ""

            if _should_add_const_ref_to_typed_param(arg.lhs, cx):
                automatic_const_part = "const "
                automatic_ref_part = "&"

            if should_add_outer_const:

                if (class_def := cx.lookup_class(arg.lhs.declared_type)) and class_def.is_unique:
                    # don't add const to unique managed param (it will impede automatic std::move from last use)
                    pass
                else:
                    automatic_const_part = "const "

            type_part = automatic_const_part + codegen_type(arg.lhs, arg_type, cx) + automatic_ref_part

            return type_part, arg.lhs.name + " = " + codegen_node(arg.rhs, cx)

        elif isinstance(arg.rhs, ListLiteral):
            if not arg.rhs.args:
                return "const std::vector<" + vector_decltype_str(arg, cx) + ">&", arg.lhs.name + " = {" + ", ".join( [codegen_node(a, cx) for a in arg.rhs.args]) + "}"
            else:
                # the above (our own poor reimplementation of CTAD with a bit of extra forward type inference) works but we can just use CTAD:

                # c++ gotcha - not usable as a default argument!
                # params.append("const auto& " + arg.lhs.name + " = std::vector {" + ", ".join(
                #         [codegen_node(a, cx) for a in arg.rhs.args]) + "}")

                # inferred part still relies on CTAD:
                vector_part = "std::vector {" + ", ".join(
                    [codegen_node(a, cx) for a in arg.rhs.args]) + "}"

                # but it's now usable as a default argument:
                return "const decltype(" + vector_part + ")& ", arg.lhs.name + " = " + vector_part
        # elif isinstance(arg.rhs, Call) and arg.rhs.func.name == "lambda":
        #     pass
        else:
            # note that this adds const& to lhs for known class constructor calls e.g. Foo() but note e.g. even if Foo() + 1 returns Foo, no automagic const& added
            if isinstance(arg.rhs, Call) and (class_def := cx.lookup_class(arg.rhs.func)):
                # it's a known class (object derived). auto add const&
                # (though std::add_const_t works here, we can be direct)
                make = codegen_node(arg.rhs, cx)
                # (we can also be direct with the decltype addition as well though decltype_str works now)
                # TODO needs tests
                ref_part = ""
                if not class_def.is_unique:
                    ref_part = "&"
                return "const decltype(" + make + ")" + ref_part, arg.lhs.name + " = " + make
            else:
                # untyped default value

                automatic_const_part = ""
                automatic_ref_part = ""
                if isinstance(arg.rhs, (ListLiteral, StringLiteral)):
                    automatic_const_part = "const "
                    automatic_ref_part = "& "

                if should_add_outer_const:
                    automatic_const_part = "const "

                type_part = automatic_const_part + "decltype(" + codegen_node(arg.rhs, cx) + ")" + automatic_ref_part

                return type_part, arg.lhs.name + " = " + codegen_node(arg.rhs, cx)
        # else:
            #     raise SemanticAnalysisError(
            #         "Non identifier left hand side in def arg", arg)
            #     # params.append("const " + decltype_str(arg.rhs, cx) + "& " + codegen_node(arg.lhs, cx) + " = " + codegen_node(arg.rhs, cx))

    return None


def _shared_ptr_str_for_type(type_node, cx):

    if not isinstance(type_node, Identifier):
        return None

    if classdef := cx.lookup_class(type_node):
        shared_ptr_str_begin = "ceto::propagate_const<std::shared_ptr<<"
        shared_ptr_str_end = ">>"
        # unique_ptr could use std::experimental::propagate_const but needs autoderef handling in ceto.h:
        unique_ptr_str_begin = "ceto::propagate_const<std::unique_ptr<<"
        unique_ptr_str_end = ">>"

        if isinstance(classdef, InterfaceDefinition):
            # TODO this clearly needs a revisit (or just scrap current 'interface' handling)
            return shared_ptr_str_begin, shared_ptr_str_end

        if classdef.is_struct:
            return None
        elif classdef.is_unique:
            return unique_ptr_str_begin, unique_ptr_str_end
        else:
            return shared_ptr_str_begin, shared_ptr_str_end

    return None


def _codegen_extern_C(lhs, rhs):
    if isinstance(lhs, Identifier) and isinstance(rhs, StringLiteral) and lhs.name == "extern" and rhs.str == "C":
        return 'extern "C"'
    return None


def _sublist_indices(sublist, lst : typing.List):
    list_set = set(lst)
    subset = set(sublist)
    if subset <= list_set:
        indices = sorted([lst.index(s) for s in subset])
        if indices == list(range(indices[0], len(indices))):
            return indices
    return None


def _propagate_const_str(string: str) -> str:
    return "ceto::propagate_const<" + string + ">"


def _codegen_compound_class_type(types, cx):
    # note that for better or worse, the calling code of codegen_type should
    # take care to add an outer const where necessary
    class_types = [(t, c) for (t, c) in [(t, cx.lookup_class(t)) for t in types] if c is not None]
    if len(class_types) == 0:
        return None
    class_type, class_def = class_types[0]
    class_name = class_type.name
    if len(class_types) > 1:
        raise CodeGenError("too many class types in type", class_type)
    typenames = [t.name for t in types]
    if class_def.is_unique:
        # we should perhaps remove these
        if indices := _sublist_indices(["const", class_name, "ref"], typenames):
            return "const ceto::propagate_const<std::unique_ptr<const " + class_name + ">>&", indices
        if indices := _sublist_indices([class_name, "const", "ref"], typenames):
            return "const ceto::propagate_const<std::unique_ptr<const " + class_name + ">>&", indices
    if not class_def.is_struct and ("ref" in typenames or "ptr" in typenames or "rref" in typenames):
        raise CodeGenError("no ref/ptr specifiers allowed for class. Use Foo.class instead, or make your class a struct", types[0])
    if indices := _sublist_indices(["shared", "mut", class_name], typenames):
        return _propagate_const_str("std::shared_ptr<" + class_name + ">"), indices
    if indices := _sublist_indices(["shared", "const", class_name], typenames):
        return _propagate_const_str("std::shared_ptr<const " + class_name + ">"), indices
    if indices := _sublist_indices(["unique", "mut", class_name], typenames):
        return _propagate_const_str("std::unique_ptr< " + class_name + ">"), indices
    if indices := _sublist_indices(["unique", "const", class_name], typenames):
        return _propagate_const_str("std::unique_ptr<const " + class_name + ">"), indices
    if indices := _sublist_indices(["weak", "mut", class_name], typenames):
        if class_def.is_unique:
            raise CodeGenError("no weak specifier for type", types[0])
        return _propagate_const_str("std::weak_ptr<" + class_name + ">"), indices
    if indices := _sublist_indices(["weak", "const", class_name], typenames):
        if class_def.is_unique:
            raise CodeGenError("no weak specifier for type", types[0])
        return _propagate_const_str("std::weak_ptr<const " + class_name + ">"), indices
    if indices := _sublist_indices(["weak", class_name], typenames):
        if class_def.is_unique:
            raise CodeGenError("no weak specifier for type", types[0])
        if mut_by_default:
            return _propagate_const_str("std::weak_ptr<" + class_name + ">"), indices
        return _propagate_const_str("std::weak_ptr<const " + class_name + ">"), indices
    if class_def.is_struct:
        # let the defaults of codegen_type handle this
        return None
    if indices := _sublist_indices(["mut", class_name], typenames):
        if class_def.is_unique:
            return _propagate_const_str("std::unique_ptr<" + class_name + ">"), indices
        return _propagate_const_str("std::shared_ptr<" + class_name + ">"), indices
    if indices := _sublist_indices(["const", class_name], typenames):
        if class_def.is_unique:
            return _propagate_const_str("std::unique_ptr<const " + class_name + ">"), indices
        return _propagate_const_str("std::shared_ptr<const " + class_name + ">"), indices
    if mut_by_default:
        if class_def.is_unique:
            return _propagate_const_str("std::unique_ptr<" + class_name + ">"), [types.index(class_type)]
        return _propagate_const_str("std::shared_ptr<" + class_name + ">"), [types.index(class_type)]
    else:
        if class_def.is_unique:
            return _propagate_const_str("std::unique_ptr<const " + class_name + ">"), [types.index(class_type)]
        return _propagate_const_str("std::shared_ptr<const " + class_name + ">"), [types.index(class_type)]


def codegen_type(expr_node, type_node, cx):

    if isinstance(expr_node, (ScopeResolution, AttributeAccess)) and type_node.name == "using":
        pass
    elif not isinstance(expr_node, (ListLiteral, TupleLiteral, Call, Identifier, TypeOp, AttributeAccess, Template)):
        raise CodeGenError("unexpected typed expression", expr_node)
    if isinstance(expr_node, Call) and not is_call_lambda(expr_node) and expr_node.func.name != "def":
        raise CodeGenError("unexpected typed call", expr_node)

    types = type_node_to_list_of_types(type_node)
    type_names = [t.name for t in types]

    if type_names[0] in ["ptr", "ref", "rref"]:
        raise CodeGenError(f"Invalid specifier. '{type_node.name}' can't be used at the beginning of a type. Maybe you want: 'auto:{type_node.name}':", type_node)

    # TODO better ptr handling - 'same behavior as add_const_t' not workable.
    # maybe just require 'unsafe' as a type keyword (not precluding 'unsafe' blocks) but
    # leave the ptr type as mut by default (needs fixes elsewhere unfortunately)

    # TODO don't make ref:ref const by default - gotcha (ideally error on a const ref:ref but not a mut ref:ref of a ptr to const etc)

    type_code = []

    if result := _codegen_compound_class_type(types, cx):
        ptr_code, indices = result
        types[indices[0]] = ptr_code  # save for below
        del indices[0]
        if indices:
            # remove the rest
            types = types[0:indices[0]] + types[indices[-1] + 1:]

    i = 0
    while i < len(types):
        t = types[i]

        assert not isinstance(t, TypeOp)

        if isinstance(t, str):  # already computed above
            code = t
        elif i < len(types) - 1 and (extern_c := _codegen_extern_C(types[i], types[i + 1])):
            type_code.append(extern_c)
            i += 2
            continue
        elif i < len(types) - 1 and type_names[i] == "ref" == type_names[i + 1]:
            type_code.append("&&")
            i += 2
            continue
        elif i < len(types) - 1 and type_names[i] == "requires":
            type_code.append("requires")
            type_code.append("(" + codegen_type(expr_node, types[i + 1], cx) + ")")
            i += 2
            continue
        elif isinstance(t, ListLiteral):
            if len(t.args) != 1:
                raise CodeGenError("Array literal type must have a single argument (for the element type)", expr_node)
            code = "std::vector<" + codegen_type(expr_node, t.args[0], cx) + ">"
        elif isinstance(t, TupleLiteral):
            if len(t.args) == 0:
                raise CodeGenError("No empty tuples as types", expr_node)
            code = "std::tuple<" + ", ".join([codegen_type(expr_node, a, cx) for a in t.args]) + ">"
        elif t.name == "ptr":
            code = "*"
        elif t.name == "ref":
            code = "&"
        elif t.name == "rref":
            code = "&&"
        elif t.name == "mut":
            raise CodeGenError("unexpected placement of 'mut'", expr_node)

        elif t.name in ["new", "goto"]:
            raise CodeGenError("nice try", t)
        elif not isinstance(t, (Identifier, Call, Template, AttributeAccess, ScopeResolution, BinOp, UnOp)):
            raise CodeGenError("unexpected type", t)
        else:
            if t.declared_type:
                temp = t.declared_type
                t.declared_type = None
                code = codegen_node(t, cx)
                t.declared_type = temp
            else:
                code = codegen_node(t, cx)

        type_code.append(code)
        i += 1

    assert len(type_code) > 0
    return " ".join(type_code)


def _structured_binding_unpack_from_tuple(node: TupleLiteral, is_for_loop_iter, cx):
    assert isinstance(node, TupleLiteral)

    structured_binding = "[" + ", ".join(codegen_node(a, cx) for a in node.args) + "]"

    if is_for_loop_iter:
        ref_part = "& "
    else:
        ref_part = " "

    if node.declared_type:
        if node.declared_type.name == "mut":
            # plain mut means by value
            return "auto " + structured_binding
        elif node.declared_type.name == "const":
            # plain const means by const value unless it's a for iter unpacking (const ref)
            return "const auto" + ref_part + structured_binding
        binding_types = type_node_to_list_of_types(node.declared_type)
        _ensure_auto_or_ref_specifies_mut_const(binding_types)
        binding_type_names = [t.name for t in binding_types]
        # we don't have to worry about stripping out a mut/const that belongs to
        # const:Foo where Foo is a ceto class (only cvref qualified 'auto'
        # allowed in structured bindings, otherwise C++ error).
        binding_types = [t for t in binding_types if t.name != "mut"]
        binding_type = list_to_typed_node(binding_types)
        binding_type_code = codegen_type(node, binding_type, cx)
        return binding_type_code + " " + structured_binding

    return "const auto" + ref_part + structured_binding


def codegen_assign(node: Assign, cx: Scope):
    assert isinstance(node, Assign)

    if not isinstance(node.lhs, (Identifier, TupleLiteral)):
        return codegen_node(node.lhs, cx) + " = " + codegen_node(node.rhs, cx)

    is_lambda_rhs_with_return_type = False

    const_specifier = ""
    if not cx.in_function_body and not cx.in_class_body:
        # add constexpr to all global defns
        constexpr_specifier = "constexpr "
    else:
        constexpr_specifier = ""

    if node.declared_type is not None and isinstance(node.rhs,
                                                     Call) and node.rhs.func.name == "lambda":
        # TODO lambda return types need fixes? fixed now?
        lambdaliteral = node.rhs
        is_lambda_rhs_with_return_type = True
        # type of the assignment (of a lambda literal) is the type of the lambda not the lhs
        if lambdaliteral.declared_type is not None:
            raise CodeGenError("Two return types defined for lambda:",
                               node.rhs)
        lambdaliteral.declared_type = node.declared_type
        newcx = cx.enter_scope()
        newcx.in_function_param_list = True
        rhs_str = codegen_lambda(lambdaliteral, newcx)
    #elif node.lhs.declared_type is None and isinstance(node.rhs, ListLiteral) and not node.rhs.args and node.rhs.declared_type is None and (vds := vector_decltype_str(node, cx)) is not None:
    elif isinstance(node.rhs, ListLiteral) and not node.rhs.args and not node.rhs.declared_type and (not node.lhs.declared_type or not isinstance(strip_mut_or_const(node.lhs.declared_type), ListLiteral)) and (vds := vector_decltype_str(node, cx)) is not None:  # TODO const/mut specifier for list type on lhs
        # handle untyped empty list literal by searching for uses
        rhs_str = "std::vector<" + vds + ">()"
    else:
        rhs_str = codegen_node(node.rhs, cx)

    if isinstance(node.lhs, Identifier):
        if _is_unique_var(node.lhs, cx):
            # ':unique' instances are unique_ptr<const T> by default but the actual variable is non-const to facilitate easier
            # automatic std::move from last use even with const:UniqueFoo (and not just mut:UniqueFoo).
            # Note that a const shared_ptr<const T> is still powerful because it can be copied from,
            # a const unique_ptr<const Foo> is rather useless (can only be called, can't form a vector with one).
            # A (not const) unique_ptr<const Foo> on the other hand is a good replacement for a
            # const shared_ptr<const Foo> (if you don't need shared ownership).
            if node.lhs.declared_type and node.lhs.declared_type.name not in ["mut", "const"]:
                return constexpr_specifier + codegen_type(node.lhs, node.lhs.declared_type, cx) + " = " + rhs_str
            else:
                old_type = node.lhs.declared_type  #  just mut or const
                node.lhs.declared_type = None
                if cx.in_class_body:
                    assign_str = "decltype(" + rhs_str + ") " + codegen_node(node.lhs, cx) + " = " + rhs_str
                else:
                    assign_str = codegen_node(node.lhs, cx) + " = " + rhs_str
                    if not node.lhs.scope.find_def(node.lhs):
                        # just auto not const auto
                        assign_str = "auto " + assign_str
                node.lhs.declared_type = old_type
                return constexpr_specifier + assign_str

        lhs_str = node.lhs.name

        if node.lhs.declared_type:

            if node.lhs.declared_type.name in ["using", "namespace"]:
                return node.lhs.declared_type.name + " " + lhs_str + " = " + rhs_str

            if node.lhs.declared_type.name == "weak":
                is_weak_const = True

            types = type_node_to_list_of_types(node.lhs.declared_type)
            type_names = [t.name for t in types]

            if any(isinstance(t, Template) and t.func.name == "template" for t in types):
                # variable template - TODO should we auto add constexpr?
                return codegen_type(node.lhs, node.lhs.declared_type, cx) + " " + lhs_str + " = " + rhs_str

            _ensure_auto_or_ref_specifies_mut_const(types)

            adjacent_type_names = zip(type_names, type_names[1:])
            if ("const", "weak") in adjacent_type_names:
                pass

            # add const if not mut

            if isinstance(node.lhs.declared_type, Identifier) and node.lhs.declared_type.name in ["mut", "const"]:
                if cx.in_class_body:
                    # lhs_type_str = "std::remove_cvref_t<decltype(" + rhs_str + ")>"  # this would work but see error message
                    raise CodeGenError(f"const data members in C++ aren't very useful and prevent moves leading to unnecessary copying. Just use {node.lhs.name} = whatever (with no {node.lhs.declared_type.name} \"type\" specified)")
                else:
                    lhs_type_str = "const auto" if node.lhs.declared_type.name == "const" else "auto"
            else:
                if cx.in_class_body:
                    # don't call 'codegen_variable_declaration_type' which adds 'const' (TODO rethink previous refactors with codegen_assign called by codegen_class). See comments elsewhere re const data members in C++.
                    lhs_type_str = codegen_type(node.lhs, node.lhs.declared_type, cx)
                    const_specifier = ""
                else:
                    const_specifier, lhs_type_str = codegen_variable_declaration_type(node.lhs, cx)

            decl_str = lhs_type_str + " " + lhs_str

            const_specifier = constexpr_specifier + const_specifier

            plain_initialization = decl_str + " = " + rhs_str

            if isinstance(node.rhs, BracedLiteral):
                # aka "copy-list-initialization" in this case
                return const_specifier + plain_initialization

            # prefer brace initialization to disable narrowing conversions (in these assignments):

            direct_initialization = lhs_type_str + " " + lhs_str + " { " + rhs_str + " } "

            # ^ but there are still cases where this introduces 'unexpected' aggregate initialization
            # e.g. l2 : std.vector<std.vector<int>> = 1
            # should it be printed as std::vector<std::vector<int>> l2 {1} (fine: aggregate init) or std::vector<std::vector<int>> l2 = 1  (error) (same issue if e.g. '1' is replaced by a 1-D vector of int)
            # I think the latter behaviour for aggregates is less suprising: = {1} can be used if the aggregate init is desired.

            if node.lhs.declared_type.name == "mut" or any(t for t in type_node_to_list_of_types(node.lhs.declared_type) if t.name == "auto"):  # or _strip_non_class_non_plain_mut_type(node.lhs, cx):
                # no need to assert non-narrowing if lhs type codegen has 'auto'
                return const_specifier + direct_initialization

            # printing of UnOp is currently parenthesized due to FIXME current use of pyparsing infix_expr discards parenthesese in e.g. (&x)->foo()   (the precedence is correct but whether explicit non-redundant parenthesese are used is discarded)
            # this unfortunately can introduce unexpected use of overparenthesized decltype (this may be a prob in other places although note use of remove_cvref etc in 'list' type deduction.
            # FIXME: this is more of a problem in user code (see test changes). Also, current discarding of RedundantParens means user code can't explicitly call over-parenthesized decltype)
            # rhs_str = re.sub(r'^\((.*)\)$', r'\1', rhs_str)

            if isinstance(node.rhs, (IntegerLiteral, FloatLiteral)) or (
                    isinstance(node.rhs, Identifier) and node.rhs.name in [
                "true", "false"]):
                return f"{const_specifier}{direct_initialization}; static_assert(std::is_convertible_v<decltype({rhs_str}), decltype({node.lhs.name})>)"

            # So go given the above, define our own no-implicit-conversion init (without the gotcha for aggregates from naive use of brace initialization everywhere). Note that typed assignments in non-block / expression context will fail on the c++ side anyway so extra statements tacked on via semicolon is ok here.

            # note that 'plain_initialization' will handle cvref mismatch errors!
            return f"{const_specifier}{plain_initialization}; static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype({rhs_str}), std::remove_cvref_t<decltype({node.lhs.name})>>)"
    elif isinstance(node.lhs, TupleLiteral):
        is_tie = False

        for a in node.lhs.args:
            if not isinstance(a, Identifier) or a.scope.find_def(a):
                if node.lhs.declared_type:
                    raise CodeGenError('typed tuple unpacking ("structured bindings" in C++) can\'t redefine variable: ', a)
                is_tie = True
            if a.declared_type:
                raise CodeGenError("unexpected type in tuple", node)

        if is_tie:
            return "std::tie(" + ", ".join(codegen_node(a, cx) for a in node.lhs.args) + ") = " + rhs_str

        structured_binding = _structured_binding_unpack_from_tuple(node.lhs, False, cx)
        return structured_binding + " = " + rhs_str

    else:
        lhs_str = codegen_node(node.lhs, cx)

    assign_str = " ".join([lhs_str, node.op, rhs_str])

    # if not hasattr(node, "already_declared") and find_def(node.lhs) is None:
    # NOTE 'already_declared' is kludge only for 'noscope' ifs
    lhs_def = node.scope.find_def(node.lhs)
    if not hasattr(node, "already_declared") and lhs_def is None:
        if cx.in_class_body:
            # "scary" may introduce ODR violation (it's fine or at least no more dangerous than C++ with explicit use of decltype in a function param list...)
            # see https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3897.html
            assign_str = "decltype(" + rhs_str + ") " + lhs_str + " = " + rhs_str
        else:
            assign_str = "const auto " + assign_str
    elif isinstance(lhs_def, FieldDefinition) and cx.in_function_body:
        # Shadow the field with a local. Use e.g. self.x = 1 to assign to the data member.
        assign_str = "const auto " + assign_str
    #elif isinstance(lhs_def, LocalVariableDefinition) and lhs_def.defining_node is node:
    elif isinstance(lhs_def, GlobalVariableDefinition):
        # TODO we should disallow shadowing ceto defined globals in function parameters and fields
        raise CodeGenError(f"{lhs_str} is a global variable defined in ceto code (which is constexpr). No shadowing or reassignment allowed.", node.lhs)

    const_specifier = constexpr_specifier + const_specifier

    return const_specifier + assign_str


def _has_outer_double_parentheses(s: str):
    """ helper to avoid acidentally emitting e.g. decltype((x + y)) which means something very different than decltype(x + y)"""

    count = 0
    ended = False
    began = False

    for char in s:
        if char == '(':
            count += 1
            if count == 2:
                began = True
        elif char == ')':
            count -= 1
            if count < 0:
                return False
            elif count == 1:
                ended = True
        if char != "(" and not began and count < 2:
            # string does not begin with at least two "("
            return False
        if ended and char != ")":
            return False

    return count == 0


def codegen_call(node: Call, cx: Scope):
    assert isinstance(node, Call)

    if is_call_lambda(node):
        newcx = cx.enter_scope()
        newcx.in_function_param_list = True
        return codegen_lambda(node, newcx)

    def ban_derefable(expr_str: str):
        # while this would make it more difficult to write e.g. thin wrappers around std.begin/std.end and std.find (allowing UB),
        # applying it to every "ordinary" Call will slow down C++ compilation and uglify the output c++ too much
        return expr_str
        if node.func.name == "defined":
            return expr_str
        capture = "&" if cx.in_function_body else ""
        # it would be nicer to write this as a c-macro but it results in compilation probs with nested invocations / no recursive expansion 
        # (e.g. when this is applied to the very large immediately invoked lambda at ast.ctp)
        return "[" + capture + "]() -> decltype(auto) { static_assert(true || !ceto::is_raw_dereferencable<std::remove_cvref_t<decltype(" + expr_str + ")>>); return " + expr_str + "; }()"

    ban_derefable_start = ""
    ban_derefable_end = ""

    if isinstance(node.func, Identifier):
        func_name = node.func.name
        if func_name == "if":
            return codegen_if(node, cx)
        elif func_name == "def":
            return codegen_def(node, cx)
        elif func_name == "operator" and len(node.args) == 1 and isinstance(
                operator_name_node := node.args[0], StringLiteral):
            return "operator" + operator_name_node.str
        elif func_name == "include":
            cpp_name = node.args[0]
            assert isinstance(cpp_name, Identifier)
            return f'#include "{cpp_name.name}.donotedit.autogenerated.h"\n'
        elif func_name == "throw":
            if not len(node.args) == 1:
                raise CodeGenError("throw takes 1 arg", node)
            return "throw " + codegen_node(node.args[0], cx)
        elif func_name == "requires":
            assert len(node.args) > 0
            block = node.args[-1]
            assert isinstance(block, Block)
            args = node.args[0:-1]
            param_strs = []

            for a in args:
                param_str = codegen_typed_def_param(a, cx)
                if not param_str and isinstance(a, Identifier):
                    param_str = codegen_node(a, cx)
                if not len(param_str) > 0:
                    raise CodeGenError("bad requires expression param", a)
                param_strs.append(param_str)

            return "requires (" + ", ".join(param_strs) + ") {" + codegen_block(block, cx) + "}"
        elif func_name in ["asinstance", "isinstance"]:
            if not len(node.args) == 2:
                raise CodeGenError("asinstance takes 2 args", node)
            class_name = node.args[1]
            const_specifier = "const "
            if extracted := extract_mut_or_const(class_name):
                mc, class_name = extracted
                if mc.name == "mut":
                    const_specifier = ""
            if not isinstance(class_name, Identifier):
                raise CodeGenError("bad is/asinstance class arg", node)
            if not (classdef := cx.lookup_class(class_name)) or classdef.is_struct or not classdef.is_concrete:
                if classdef.is_concrete:
                    raise CodeGenError("is/asinstance arg can't be a template", classdef)
                raise CodeGenError("is/asinstance arg must be a class", node)
            cast_string = "std::dynamic_pointer_cast<" + const_specifier + class_name.name + ">(" + codegen_node(node.args[0], cx) + ")"
            if func_name == "isinstance":
                cast_string = "(" + cast_string + " != nullptr)"
            return cast_string
        elif func_name == "namespace":
            if len(node.args) == 0:
                raise CodeGenError("empty namespace args", node)
            block = node.args[-1]
            if not isinstance(block, Block):
                raise CodeGenError("namespace last args must be a Block", node)
            if len(node.args) > 2:
                raise CodeGenError("too many namespace args", node)
            if len(node.args) == 2:
                name = node.args[0]
                if not isinstance(name, (Identifier, ScopeResolution, AttributeAccess)):
                    raise CodeGenError("bad namespace name", node)
                name_code = codegen_node(name, cx)
            else:
                name_code = ""
            block_code = codegen_block(block, cx.enter_scope())
            return "namespace " + name_code + " {\n" + block_code + "\n}"
        elif func_name == "defmacro":
            return "\n"
        elif func_name == "unsafe":
            assert len(node.args) == 0  # enfored in sema
            if not isinstance(node.parent, Block):
                raise CodeGenError("unsafe() call must be in Block", node)
            #if node.parent.args[0] is not node:
            #    raise CodeGenError("unsafe() call must be first statement in Block", node)
            cx.is_unsafe = True
            return "// unsafe"
        elif func_name == "ceto_private_module_boundary" and len(node.args) == 0:
            cx.is_unsafe = False
            return "\n"
        elif func_name == "overparenthesized_decltype" and len(node.args) == 1:
            # calling plain "decltype" in ceto will always strip outer double parenthesese 
            # (they are often accidentally added by codegen's overeager parenthesization)
            return "decltype((" + codegen_node(node.args[0], cx) + "))"
        else:
            old_unsafe = None

            if func_name == "static_assert":
                old_unsafe = cx.is_unsafe
                cx.is_unsafe = False

            arg_strs = [codegen_node(a, cx) for a in node.args]
            
            if old_unsafe is not None:
                cx.is_unsafe = old_unsafe

            args_inner = ", ".join(arg_strs)
            args_str = "(" + args_inner + ")"

            if class_def := cx.lookup_class(node.func):
                class_name = node.func.name
                curly_args = "{" + args_inner + "}"

                if not node.args:
                    # just use round parentheses to call default constructor
                    curly_args = args_str

                # if class_def.has_generic_params():
                #     class_name += "<" + ", ".join(
                #         [decltype_str(a, cx) for i, a in enumerate(node.args) if
                #          class_def.is_generic_param_index[i]]) + ">"

                if class_def.is_struct:
                    func_str = class_name
                    args_str = curly_args
                else:
                    if not class_def.is_concrete:
                        class_name = "decltype(" + class_name + curly_args + ")"

                    const_part = "const " if _is_const_make(node) else ""

                    if class_def.is_unique:
                        func_str = "ceto::make_unique_propagate_const<" + const_part + class_name + ">"
                    else:
                        func_str = "ceto::make_shared_propagate_const<" + const_part + class_name + ">"

                return func_str + args_str

            if isinstance(node.func, Identifier):
                func_str = node.func.name

                # TODO we do want to ban a number of decltype uses - but not yet
                # TODO we should verify that "defined" inside if:preprocessor (or ban both in safe mode)
                if not cx.is_unsafe and func_str not in ["decltype", "defined", "static_assert", "include"]:
                    if not node.scope.lookup_function(node.func) and not node.scope.find_def(node.func):
                        raise CodeGenError("call to unknown function - use unsafe to call external C++", node)
            else:
                # TODO handle safety check for namespaced/static calls

                func_str = codegen_node(node.func, cx)

            if cx.in_function_body:
                capture = "&"
            else:
                capture = ""

            simple_call_str = func_str + args_str

            if func_str in ("decltype", "static_assert") or (
                    isinstance(node.parent,
                               (ScopeResolution, ArrowOp, AttributeAccess)) and
                    node.parent.lhs is not node):
                if func_str == "decltype":
                    cx.in_decltype = True
                    if _has_outer_double_parentheses(args_str):
                        # likely accidental overparenthesized decltype due to overenthusiastic parenthization in codegen
                        # strip the outside ones (fine for now)
                        return func_str + args_str[1:-1]
                return simple_call_str

            if isinstance(node.parent, Template) and isinstance(node.parent.func, AttributeAccess) and node.parent.func.lhs.name == "std" and node.parent.func.rhs.name == "function":
                # TODO class constructor calls and std.function like calls e.g. std.function<int(int)> also interact poorly with ceto classes:
                # need fix for std.function<Foo()>
                return simple_call_str

            #return "CETO_BAN_RAW_DEREFERENCABLE(" + simple_call_str + ")"
            return ban_derefable(simple_call_str)

            # the below works in many cases but not with c++20 style vector CTAD. We'd have to go back to our py14 style diy vector CTAD to allow call_or_construct code in a vector e.g. ceto code of form l = [Foo(), Foo(), Foo()]

            dt_str = "decltype(" + simple_call_str + ")"

            simple_return = "return "
            if isinstance(node.parent, Block):
                simple_return = ""

            call_str = "[" + capture + "] { if constexpr (std::is_base_of_v<ceto::object, " + dt_str + ">) { return ceto::call_or_construct<" + dt_str + ">" + args_str + "; } else { " + simple_return + simple_call_str + "; } } ()"

            return call_str
    else:
        # probably a method:

        # not auto-flattening args is becoming annoying!
        # TODO make bin-op arg handling more ergonomic - maybe flatten bin-op args earlier (look at e.g. sympy's Add/Mul handling)
        method_name = None
        method_target = None
        if isinstance(node.func, (AttributeAccess, ScopeResolution)):
            method_name = node.func.rhs
            method_target = node.func.lhs
            while isinstance(method_name, (AttributeAccess, ScopeResolution)):
                method_name = method_name.rhs

        func_str = None
        new_func = node.func

        if method_name is not None:
            if not cx.is_unsafe and method_name.name in ["begin", "cbegin", "end", "cend", "rbegin", "rend", "crbegin", "crend", "data", "c_str", "find", "lower_bound", "upper_bound"]:
                # TODO ban equal_range for std::map etc
                # Note that e.g. this effectively bans emplace and insert (with iterator args) for std::vector
                ban_derefable_start = "CETO_BAN_RAW_DEREFERENCABLE("
                ban_derefable_end = ")"

            def consume_method_name():
                method_parent = method_name.parent
                assert method_parent.rhs is method_name

                if method_parent in method_parent.parent.args:
                    na = list(method_parent.parent.args)
                    na.remove(method_parent)
                    na.append(method_parent.lhs)
                    return AttributeAccess(".", na)
                elif method_parent is method_parent.parent.func:
                    return method_parent.lhs

            if method_name.name == "operator" and len(
                    node.args) == 1 and isinstance(operator_name_node := node.args[0], StringLiteral):
                new_func = consume_method_name()
                return "(*ceto::mad(" + codegen_node(new_func, cx) + ")).operator" + operator_name_node.str

            elif method_name.parent and not isinstance(method_name.parent,
                                                       (ScopeResolution, ArrowOp)):
                # method_name.parent is None for a method call inside a decltype in a return type
                # TODO we maybe still have to handle cases where method_name.parent is None. silly example: x : decltype([].append(1)[0])

                new_func = consume_method_name()

                if isinstance(node.func, AttributeAccess) and method_name.name == "append" and len(node.args) == 1:
                    append_target_lhs = method_target
                    while isinstance(append_target_lhs, BinOp):
                        append_target_lhs = append_target_lhs.lhs

                    if append_target_lhs.name in ["this", "self"] or isinstance(node.scope.find_def(append_target_lhs), VariableDefinition):
                        # perhaps controversial rewriting of append to push_back
                        # this would also be the place to e.g. disable all unsafe std::vector methods (require a construct like (&my_vec)->data() to workaround while signaling unsafety)
                        is_list = False
                        if isinstance(new_func, ListLiteral):
                            is_list = True
                        else:
                            for d in node.scope.find_defs(new_func):
                                if isinstance(d.defining_node, Assign) and isinstance(
                                        d.defining_node.rhs, ListLiteral):
                                    is_list = True
                                    break

                        if is_list:
                            return "(" + codegen_node(new_func, cx) + ").push_back(" + codegen_node(node.args[0], cx) + ")"
                        else:
                            # we still provide .append as .push_back for all std::vectors even in generic code
                            # (note this function performs an autoderef on new_func):
                            #return "CETO_BAN_RAW_DEREFERENCABLE(ceto::append_or_push_back(" + codegen_node(new_func, cx) + ", " + codegen_node(node.args[0], cx) + "))"
                            return ban_derefable("ceto::append_or_push_back(" + codegen_node(new_func, cx) + ", " + codegen_node(node.args[0], cx) + ")")

                new_attr_access = AttributeAccess(".", [new_func, method_name])
                new_attr_access.parent = node
                func_str = codegen_attribute_access(new_attr_access, cx)

        if func_str is None:
            func_str = codegen_node(new_func, cx)

        #return "CETO_BAN_RAW_DEREFERENCABLE(" + func_str + "(" + ", ".join(
        #    map(lambda a: codegen_node(a, cx), node.args)) + "))"
        #return ban_derefable(func_str + "(" + ", ".join(
        #    map(lambda a: codegen_node(a, cx), node.args)) + ")")
        return ban_derefable_start + func_str + "(" + ", ".join(
            map(lambda a: codegen_node(a, cx), node.args)) + ")" + ban_derefable_end


def _is_const_make(node : Call):
    assert isinstance(node, Call)

    is_const = not mut_by_default

    if node.declared_type is not None:
        lhs_type = node.declared_type

        if isinstance(lhs_type, Identifier):
            if lhs_type.name == "mut":
                is_const = False
            elif lhs_type.name == "const":
                is_const = True
        elif isinstance(lhs_type, TypeOp):
            type_list = type_node_to_list_of_types(lhs_type)
            if "mut" in type_list:
                is_const = False
            elif "const" in type_list:
                is_const = True
    elif isinstance(node.parent,
                    Assign) and node.parent.lhs.declared_type is not None:
        lhs_type = node.parent.lhs.declared_type

        if isinstance(lhs_type, Identifier):
            if lhs_type.name == "mut":
                is_const = False
        elif isinstance(lhs_type, TypeOp):
            type_list = type_node_to_list_of_types(lhs_type)
            if "mut" in [type_list[0].name, type_list[-1].name]:
                is_const = False

    return is_const


def codegen_variable_declaration_type(node: Identifier, cx: Scope):
    if not isinstance(node, Identifier):
        raise CodeGenError("Unexpected typed expression", node)
    assert node.declared_type

    lhs_type_str = None

    const_specifier = ""

    classdef = None

    if isinstance(node.declared_type, Identifier):
        if node.declared_type.name == "auto":
            raise CodeGenError("must specify const/mut for auto", node)
        classdef = cx.lookup_class(node.declared_type)
    elif isinstance(node.declared_type, TypeOp):
        type_list = type_node_to_list_of_types(node.declared_type)

        _ensure_auto_or_ref_specifies_mut_const(type_list)

        classdefs = [c for c in [cx.lookup_class(t) for t in type_list] if c is not None]

        mut_or_const = [t for t in type_list if t.name in ["mut", "const"]]

        if len(classdefs) > 1:
            raise CodeGenError("too many classes specified", node)

        if len(mut_or_const) > 1 and classdefs and not classdefs[0].is_struct:
            # note that e.g. mut:auto:const:ptr is allowed
            raise CodeGenError("too many mut/const specified for class type", node)

        if classdefs:
            classdef = classdefs[0]

        if classdef and mut_or_const:
            lhs_type_str = codegen_type(node, node.declared_type, cx)

            if mut_or_const[0].name == "const":
                lhs_type_str = "const " + lhs_type_str

        elif "mut" in [type_list[0].name,
                       type_list[-1].name]:  # east or west mut is fine
            if type_list[0].name == "mut":
                type_list.pop(0)
            else:
                type_list.pop()
            rebuilt = list_to_typed_node(type_list)
            lhs_type_str = codegen_type(node, rebuilt, cx)
            assert len(lhs_type_str) > 0
        else:
            for i, t in enumerate(type_list):
                otheridx = i - 1 if i > 0 else i + 1
                if (t.name in ["const", "auto"] and type_list[otheridx].name in ["const", "auto"]) or (
                        t.name == "const" and i < len(type_list) - 1 and type_list[i + 1].name == "ref"):
                    # either contains "const auto", "auto const" or contains "const const"/"auto auto" (error in c++)
                    # alternately contains "const ref" anywhere
                    # use type verbatim
                    lhs_type_str = codegen_type(node, node.declared_type, cx)
                    break

            if lhs_type_str is None:
                if (type_list[0].name == "const" and type_list[-1].name != "ptr") or type_list[-1].name == "const":
                    lhs_type_str = codegen_type(node, node.declared_type, cx)
                elif type_list[-1].name == "ptr":
                    lhs_type_str = codegen_type(node, node.declared_type, cx) + " const"
                # else:
                #     assert 0

    if lhs_type_str is None:
        lhs_type_str = codegen_type(node, node.declared_type, cx)
        needs_const = not mut_by_default
        if needs_const and not const_specifier and not (classdef and classdef.is_unique):
            const_specifier = "const "

    return const_specifier, lhs_type_str



def decltype_str(node, cx):
    if isinstance(node, ArrayAccess):

        # for n, c in cx.find_defs(node.func): # doesn't work for forward inference (would require 2 passes - just keep using old find_defs for now)
        # for n, c in find_defs(node.func):
        for d in node.scope.find_defs(node.func):
            c = d.defining_node
            if isinstance(c, Assign):# and hasattr(c.rhs, "_element_decltype_str"):
                if vds := vector_decltype_str(c, cx):
                    return vds

        return "std::remove_cvref_t<{}>::value_type".format(decltype_str(node.func, cx))

    elif isinstance(node, ListLiteral):

        return "std::vector<" + decltype_str(node.args[0], cx) + ">"

    else:

        needs_outer_decltype, code = _decltype_str(node, cx)
        if needs_outer_decltype:
            code = "decltype(" + code + ")"
        return code


def _decltype_maybe_wrapped_in_declval(node, cx):
    needs_decltype, code = _decltype_str(node, cx)
    if not needs_decltype:
        return "std::declval<" + code + ">()"
    return code


def _decltype_str(node, cx):

    if isinstance(node, (IntegerLiteral, FloatLiteral, StringLiteral)):
        return True, codegen_node(node, cx)

    if isinstance(node, BinOp):
        binop = node
        if isinstance(node, AttributeAccess):
            # TODO needs fixes for implicit scope resolution / may be busted
            return True, "(*ceto::mad(" + _decltype_maybe_wrapped_in_declval(node.lhs, cx) + "))." + codegen_node(node.rhs, cx)
        elif isinstance(node, ArrowOp):
            return True, _decltype_maybe_wrapped_in_declval(node.lhs, cx) + "->" + codegen_node(node.rhs, cx)

        return True, _decltype_maybe_wrapped_in_declval(binop.lhs, cx) + binop.op + _decltype_maybe_wrapped_in_declval(binop.rhs, cx)

    elif isinstance(node, UnOp):
        assert False, "this needs fixes"
        return True, "(" + node.op + _decltype_str(node.args[0], cx)[1] + ")"  # the other place unop is parenthesized is "necessary". here too?
    elif isinstance(node, Call):

        if node.func.name == "lambda":
            # this isn't going to work if the lambda captures variables (at least variables defined after the empty list whose type we're creating).
            # lambdas with a capture list are easier to handle - TODO move the implicit capture to an earlier pass than codegen.
            return True, codegen_lambda(node, cx)

        call = node

        if class_def := cx.lookup_class(node.func):
            class_name = node.func.name

            # if class_def.has_generic_params():
            #     class_name += "<" + ", ".join(
            #         [decltype_str(a, cx) for i, a in enumerate(node.args) if class_def.is_generic_param_index[i]]) + ">"

            # instead of manual tracking like the above,
            # leave the matter of the desired class type up to C++ CTAD:
            args_str = "{" + ", ".join([_decltype_maybe_wrapped_in_declval(a, cx) for a in node.args]) + "}"
            if not node.args:
                args_str = "()"  # use round parens instead of curlies for default case

            const = "const " if _is_const_make(call) else ""

            class_name = "decltype(" + class_name + args_str + ")"

            if class_def.is_struct:
                func_str = class_name
            elif class_def.is_unique:
                func_str = "std::unique_ptr<" + const + class_name + ">"
            else:
                func_str = "std::shared_ptr<" + const + class_name + ">"

            return False, func_str

        else:
            # return True, codegen_node(call, cx)
            # return True, codegen_node(call.func, cx) + "(" + ", ".join([_decltype_str(a, cx)[1] for a in call.args]) + ")"
            #return True, codegen_node(call.func, cx) + "(" + ", ".join([_decltype_maybe_wrapped_in_declval(a, cx) for a in call.args]) + ")"
            return False, "std::remove_cvref_t<decltype(" + codegen_node(call.func, cx) + "(" + ", ".join([_decltype_maybe_wrapped_in_declval(a, cx) for a in call.args]) + "))>"
    elif isinstance(node, ListLiteral):
        # return True, "std::vector<" + decltype_str(node.args[0], cx) + "> {}"
        return False, "std::vector<" + decltype_str(node.args[0], cx) + ">"
    elif isinstance(node, ArrayAccess):

        # for n, c in cx.find_defs(node.func): # doesn't work for forward inference (would require 2 passes - just keep using old find_defs for now)
        # for n, c in find_defs(node.func):
        for d in node.scope.find_defs(node.func):
            c = d.defining_node
            if isinstance(c, Assign):# and hasattr(c.rhs, "_element_decltype_str"):
                if vds := vector_decltype_str(c, cx):
                    return False, vds
        return False, "std::remove_cvref_t<{}>::value_type".format(decltype_str(node.func, cx))

    elif isinstance(node, TupleLiteral):
        return True, "std::make_tuple(" + ", ".join(_decltype_maybe_wrapped_in_declval(a, cx) for a in node.args) + ")"

    assert isinstance(node, Identifier)

    # defs = list(find_defs(node))   # fails because only 1 (uncompleted) pass over ast to build scopes
    # defs = list(cx.find_defs(node))   # fails because only 1 (uncompleted) pass over ast to build scopes
    defs = list(node.scope.find_defs(node))
    if not defs:
        return True, node.name

    eligible_defs = list(defs)

    for d in defs:
        def_node, def_context = d.defined_node, d.defining_node

        if def_node.declared_type:

            if def_node.declared_type.name in ["mut", "const", "auto", "weak", "shared", "unique"]:
                # plain mut/const/auto provides no element type info for declarations (assign rhs may inform type)
                if isinstance(def_context, Identifier):
                    eligible_defs.remove(d)
                    continue

            elif isinstance(def_node.declared_type, TypeOp):

                type_list = type_node_to_list_of_types(def_node.declared_type)

                if any(t.name == "auto" for t in type_list):
                    # any use of auto also gives us no type info
                    if isinstance(def_context, Identifier):
                        eligible_defs.remove(d)
                        continue

                elif "mut" in [type_list[0].name, type_list[-1].name]:
                    # discard east or west mut
                    if type_list[0].name == "mut":
                        type_list.pop(0)
                    else:
                        type_list.pop()
                    # there should be a real type left
                    rebuilt = list_to_typed_node(type_list)
                    return False, codegen_type(def_node, rebuilt, cx)
            else:
                return False, codegen_type(def_node, def_node.declared_type, cx)
        eligible_defs.append((def_node, def_context))

    if not eligible_defs:
        return True, node.name

    # TODO cleanup dodgy 'continue' use above
    last_def = eligible_defs[-1]
    if isinstance(last_def, tuple):
        last_ident, last_context = eligible_defs[-1]
    else:
        last_ident = eligible_defs[-1].defined_node
        last_context = eligible_defs[-1].defining_node

    assert isinstance(last_ident, Identifier)

    if isinstance(last_context, Assign) and not (isinstance(last_context.rhs, ListLiteral) and not last_context.rhs.args and not last_context.rhs.declared_type):
        return _decltype_str(last_context.rhs, cx)
    elif isinstance(last_context, Assign) and isinstance(last_context.rhs, ListLiteral):
        if vds := vector_decltype_str(last_context, cx):
            return False, "std::vector<" + vds + ">"

    elif isinstance(last_context, Call) and last_context.func.name == "for":
        instmt = last_context.args[0]
        if not isinstance(instmt, BinOp) and instmt.op == "in":
            raise CodeGenError("for loop should have in-statement as first argument ", last_context)
        if last_ident is instmt.lhs:  # maybe we should adjust find_defs to return the in-operator ?
            return False, "std::ranges::range_value_t<" + decltype_str(instmt.rhs, cx) + ">"  # broken in clang < 16 but that's ok

            # previously messed up declval/Call handling led to this unnecessary/broken declval usage.
            return True, "std::declval<typename std::remove_cvref_t<" + decltype_str(instmt.rhs, cx) + ">::value_type>()"  # only works for std::vector and similar
            # this breaks in clang14/15 (fine in 16) but that's ok - no python style empty lists for old clang users:
            return True, "std::declval<std::ranges::range_value_t<" + decltype_str(instmt.rhs, cx) + ">>()"

    else:
        # return True, codegen_node(last_ident, cx)
        return True, last_ident.name  # hope that we validated this is a legal name elsewhere...


def vector_decltype_str(node, cx):
    rhs_str = None
    assert isinstance(node, Assign)

    if isinstance(node, Assign) and isinstance(node.rhs, ListLiteral) and node.rhs.args:
        return decltype_str(node.rhs.args[0], cx)

    for found_use_node in find_uses(node):
        parent = found_use_node.parent
        while rhs_str is None and not isinstance(parent, Block):
            found_use_context = parent

            if isinstance(found_use_context, AttributeAccess) and (
               found_use_context.lhs is found_use_node and found_use_context.rhs.name == "append"):

                if isinstance(found_use_context.parent, Call) and len(found_use_context.parent.args) == 1:
                    append_arg = found_use_context.parent.args[0]
                    rhs_str = decltype_str(append_arg, cx)

            if isinstance(found_use_context, Assign) and isinstance(found_use_context.lhs, ArrayAccess) and found_use_context.lhs.func is found_use_node:
                array_element_new_val = found_use_context.rhs
                rhs_str = decltype_str(array_element_new_val, cx)

            parent = parent.parent

        if rhs_str is not None:
            break
    return rhs_str


def codegen_node(node: Node, cx: Scope):
    assert isinstance(node, Node)

    if node.declared_type is not None:
        if not isinstance(node, (ListLiteral, Call)):

            if isinstance(node, Identifier):

                if not isinstance(node.parent, Template):
                    made_easy_lambda_args_mistake = False
                    parent = node.parent
                    while parent:
                        if isinstance(parent, Call):
                            if parent.func.name == "lambda":
                                made_easy_lambda_args_mistake = True
                                break
                            elif parent.func.name in ["def", "class", "sruct"]:
                                made_easy_lambda_args_mistake = False
                                break
                        elif isinstance(parent, Block) and len(parent.args) != 1:
                            made_easy_lambda_args_mistake = False
                            break
                        parent = parent.parent
                    if made_easy_lambda_args_mistake:
                        raise CodeGenError("do you have the args wrong? [ it's lambda(x, 5) not lambda(x: 5) ] in ", parent)
            elif not isinstance(node, (AttributeAccess, Template)):
                raise CodeGenError("unexpected context for typed construct", node)

            return codegen_type(node, node, cx)  # this is a type inside a more complicated expression e.g. std.is_same_v<Foo, int:ptr>
        elif isinstance(node, Call) and node.func.name != "def" and not is_call_lambda(node) and node.declared_type.name not in ["const", "mut"]:
            raise CodeGenError("Unexpected typed call", node)

    if isinstance(node, Call):
        return codegen_call(node, cx)
    elif isinstance(node, (IntegerLiteral, FloatLiteral)):
        return str(node)
    elif isinstance(node, Identifier):
        name = node.name

        if name == "ptr":
            raise CodeGenError("Use of 'ptr' outside type context is an error", node)
        elif name == "ref":
            raise CodeGenError("Use of 'ref' outside type context is an error", node)
        elif name == "None":
            return "nullptr"
        elif name == "True":
            return "true"
        elif name == "False":
            return "false"
        elif name == "...":
            return "..."
        elif name == "string" and not isinstance(node.parent, (AttributeAccess, ScopeResolution, ArrowOp)):
            return "std::string"
        # elif name == "object":
        #     return "std::shared_ptr<object>"

        if cx.in_function_body and not (isinstance(node.parent, (AttributeAccess, ScopeResolution, ArrowOp)) and node is node.parent.rhs) and isinstance(node.scope.find_def(node), FieldDefinition):
            raise CodeGenError(f"no direct access to fields - use self.{node.name} instead of just {node.name}", node)

        if not (isinstance(node.parent, (AttributeAccess, ScopeResolution)) and
                node is node.parent.lhs) and (
           ptr_name := _shared_ptr_str_for_type(node, cx)):
            ptr_begin, ptr_end = ptr_name
            return ptr_begin + ("const " if not mut_by_default else "") + name + ptr_end

        is_last_use = is_last_use_of_identifier(node)

        if is_last_use and _is_unique_var(node, cx):
            return "std::move(" + name + ")"

        return name

    elif isinstance(node, BinOp):

        if 0 and isinstance(node, NamedParameter):
            raise SemanticAnalysisError("Unparenthesized assignment treated like named parameter in this context (you need '(' and ')'):", node)

        elif isinstance(node, TypeOp):
            if isinstance(node, SyntaxTypeOp):
                if node.lhs.name == "return":
                    ret_body = codegen_node(node.rhs, cx)
                    if node.synthetic_lambda_return_lambda is not None:
                        lambdanode = node.synthetic_lambda_return_lambda
                        assert lambdanode
                        if lambdanode.declared_type is not None:
                            if lambdanode.declared_type.name == "void":
                                # the below code works but let's avoid needless "is_void_v<void>" checks
                                return ret_body

                            declared_type_constexpr = "&& !std::is_void_v<" + codegen_type(lambdanode, lambdanode.declared_type, cx) + ">"
                        else:
                            declared_type_constexpr = ""
                        # return if not void (void cast to suppress unused value warning - [[maybe_unused]] doesn't apply to (void) expressions)
                        return "if constexpr (!std::is_void_v<decltype(" + ret_body + ")>" + declared_type_constexpr + ") { return " + ret_body + "; } else { static_cast<void>(" + ret_body + "); }"
                    else:
                        return "return " + ret_body
                else:
                    assert False
            else:
                # dealing with subtype in a compound "type" (we're considering e.g. explict function template params as just complicated "return" types)
                # (consequence of not creating a .declared_type property for nested subtypes)

                # same handling as above case for nested expression with .declared_type

                # codegen_type should be handling compound types safely (any error checking for TypeOp lhs rhs should go there)
                return codegen_type(node, node, cx)

        elif isinstance(node, Assign):
            assign_code = codegen_assign(node, cx)
            if isinstance(node.parent, (BinOp, UnOp)) \
                    and not isinstance(node.parent, SyntaxTypeOp):  # avoid overparensing an assign in one-liner if condition
                assign_code = "(" + assign_code + ")"
            return assign_code

        else:
            if isinstance(node, AttributeAccess):
                return codegen_attribute_access(node, cx)

            elif is_comment(node):
                # probably needs to go near method handling logic now that precedence issue fixed (TODO re-enable comment stashing)
                if not (len(node.rhs.args) == 1 or isinstance(node.rhs.args[0], StringLiteral)):
                    raise CodeGenError("unexpected ceto::comment ", node)
                return "//" + node.rhs.args[0].func.replace("\n", "\\n") + "\n"

            opstr = node.op
            if opstr == "and":  # don't use the weird C operators tho tempting
                opstr = "&&"
            elif node.op == "or":
                opstr = "||"

            binop_str = " ".join([codegen_node(node.lhs, cx), opstr, codegen_node(node.rhs, cx)])

            if isinstance(node.parent, (BinOp, UnOp)) and not isinstance(node.parent, (ScopeResolution, ArrowOp, AttributeAccess)):
                # guard against precedence mismatch (e.g. extra parenthesese
                # not strictly preserved in the ast)
                # untested / maybe-buggy
                binop_str = "(" + binop_str + ")"

            return binop_str

    elif isinstance(node, ListLiteral):
        list_type = node.declared_type

        if list_type is None and isinstance(node.parent, Assign) and (list_assign_type := node.parent.lhs.declared_type):
            list_assign_type = strip_mut_or_const(list_assign_type)

            if isinstance(list_assign_type, ListLiteral):
                if not len(list_assign_type.args) == 1:
                    raise CodeGenError("unexpected args in list-type", list_assign_type)
                list_type = list_assign_type.args[0]

        elements = [codegen_node(e, cx) for e in node.args]

        if list_type is not None:
            return "std::vector<{}>{{{}}}".format(codegen_type(node, list_type, cx), ", ".join(elements))
        elif elements:
            # note: [[[1,2,3,4]]] should be a vector with 1 element not 4!

            if len(elements) > 1 or isinstance(node.args[0], ListLiteral):
                return "std::vector {{" + ", ".join(elements) + "}}"
            else:
                return "std::vector {" + ", ".join(elements) + "}"

            # no longer necessary CTAD reimplementation:
            # return "std::vector<{}>{{{}}}".format(decltype_str(node.args[0], cx), ", ".join(elements))
        else:
            raise CodeGenError("Cannot create vector without elements", node)
    elif isinstance(node, BracedLiteral):
        if isinstance(node.parent, Block):
            raise CodeGenError("Curly brace expression is invalid here. Use 'scope' for an anonymous scope.", node)
        elements = [codegen_node(e, cx) for e in node.args]
        return "{" + ", ".join(elements) + "}"
    elif isinstance(node, TupleLiteral):
        elements = [codegen_node(e, cx) for e in node.args]
        return "std::make_tuple(" + ", ".join(elements) + ")"
    elif isinstance(node, ArrayAccess):
        if len(node.args) > 1:
            raise CodeGenError("advanced slicing not supported yet")
        func_str = codegen_node(node.func, cx)
        idx_str = codegen_node(node.args[0], cx)
        return "ceto::bounds_check(" + func_str + "," + idx_str + ")"
        #raise CodeGenError("Array accesses should have been lowered in a macro pass! You've probably written your own array[access] defmacro (it's buggy)", node)
    elif isinstance(node, BracedCall):
        if cx.lookup_class(node.func):
            # cut down on multiple syntaxes for same thing (even though the make_shared/unique call utilizes curly braces)
            raise CodeGenError("Use round parentheses for ceto-defined class/struct constructor call (curly braces are automatic)", node)
        return codegen_node(node.func, cx) + "{" + ", ".join(codegen_node(a, cx) for a in node.args) + "}"
    elif isinstance(node, UnOp):
        opername = node.op
        if opername == ":":
            assert 0
        elif opername == "not":
            return "!" + codegen_node(node.args[0], cx)
        else:
            return "(" + opername + codegen_node(node.args[0], cx) + ")"
            # return opername + codegen_node(node.args[0], cx)
    elif isinstance(node, LeftAssociativeUnOp):
        opername = node.op
        return codegen_node(node.args[0], cx) + opername
    elif isinstance(node, StringLiteral):
        if not (node.prefix or node.suffix): # and isinstance(node.parent, Call) and node in node.parent.args and not cx.lookup_class(node.parent.func):
            return node.escaped()  # c-string by default
        ffixes = [f.name for f in [node.prefix, node.suffix] if f]
        if "c" in ffixes and "s" in ffixes:
            raise CodeGenError("string literal cannot be both c-string and std::string", node)
        if node.prefix and node.prefix.name == "include":
            return '#include "' + node.str + '"\n'
        if node.prefix and node.prefix.name == "char":
            # TODO more error checking / needs escape of '
            return "'" + node.str + "'"
        if node.prefix and node.prefix.name == "cpp":
            if node.suffix:
                raise CodeGenError("no suffixes for cpp-string", node)
            # unsafe embedded c++
            return node.str
        if "c" in ffixes or "s" in ffixes:
            if node.prefix and node.prefix.name in ["c", "s"]:
                str_prefix = node.prefix
                node.prefix = None
                code = str(node)
                node.prefix = str_prefix
                if node.prefix.name == "s":
                    return "std::string {" + code + "}"
                else:
                    return code
            elif node.suffix and node.suffix.name in ["c", "s"]:
                str_suffix = node.suffix
                node.suffix = None
                code = str(node)
                node.suffix = str_suffix
                if node.suffix.name == "s":
                    return "std::string {" + code + "}"
                else:
                    return code
        if node.prefix or node.suffix:
            return str(node)
        return "std::string {" + str(node) + "}"
    # elif isinstance(node, RedundantParens):  # too complicated letting codegen deal with this. just disable -Wparens
    #     return "(" + codegen_node(node.args[0]) + ")"
    elif isinstance(node, Template):
        if node.func.name == "include":
            if not len(node.args) == 1:
                raise CodeGenError("bad angle include args", node)
            # note abuse of division and attribute access (in additon to template syntax) for e.g. include<pybind11/stl.h>
            return "#include <" + "".join(str(node.args[0]).split()).replace("(", "").replace(")", "") + ">\n"

        # allow auto shared_ptr etc with parameterized classes e.g. f : Foo<int> results in shared_ptr<Foo<int>> f not shared_ptr<Foo><int>(f)
        # (^ this is a bit of a dubious feature when e.g. f: decltype(Foo(1)) works without this special case logic)
        template_args = "<" + ",".join([codegen_node(a, cx) for a in node.args]) + ">"
        if ptr_name := _shared_ptr_str_for_type(node.func, cx):
            ptr_begin, ptr_end = ptr_name
            return ptr_begin + ("const " if not mut_by_default else "") + node.func.name + template_args + ptr_end
        else:
            return codegen_node(node.func, cx) + template_args

    assert False, "unhandled node: " + str(node)
