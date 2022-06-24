from typing import Union, Any

from semanticanalysis import Node, Module, Call, Block, UnOp, BinOp, ColonBinOp, Assign, NamedParameter, Identifier, IntegerLiteral, IfNode, SemanticAnalysisError, SyntaxColonBinOp, find_def, find_use
from parser import ListLiteral, TupleLiteral, ArrayAccess, StringLiteral, AttributeAccess


import io


class CodeGenError(Exception):
    pass

cpp_preamble = """
#include <memory>
#include <cstdio>

class object {
    virtual ~{
    };
};

"""


# Uses code and ideas from https://github.com/lukasmartinelli/py14

# https://brevzin.github.io/c++/2019/12/02/named-arguments/


def codegen_if(ifcall : Call):
    assert isinstance(ifcall, Call)
    assert ifcall.func.name == "if"

    ifnode = IfNode(ifcall.func, ifcall.args)

    cpp = "if (" + codegen_node(ifnode.cond) + ") {\n"
    cpp += codegen_block(ifnode.thenblock)

    for elifcond, elifblock in ifnode.eliftuples:
        cpp += "} else if (" + codegen_node(elifcond) + ") {\n"
        cpp += codegen_block(elifblock)

    if ifnode.elseblock:
        cpp += "} else {\n"
        cpp += codegen_block(ifnode.elseblock)

    cpp += "}\n"

    return cpp


def codegen_block(block: Block, indent):
    assert isinstance(block, Block)
    cpp = ""
    indent_str = "    " * indent
    for b in block.args:
        cpp += indent_str + codegen_node(b, indent) + ";\n"
        # if isinstance(b, Call):
        #     if b.func.name == "if":
        #         cpp += codegen_if(b)
        # elif isinstance(b, BinOp):
        #     pass
        # elif isinstance(b, UnOp):
        #     pass
    return cpp


# def indent(text, amount, ch=' '):
    # return textwrap.indent(text, amount * ch)


def codegen_def(defnode: Call, indent):
    assert defnode.func.name == "def"
    name = defnode.args[0].name
    args = defnode.args[1:]
    block = args.pop()
    assert isinstance(block, Block)

    params = []
    for idx, arg in enumerate(args):
        params.append(("T" + str(idx + 1), arg.name))
    typenames = ["typename " + arg[0] for arg in params]

    template = "inline "
    if name == "main":
        template = ""
    if typenames:
        template = "template <{0}>\n".format(", ".join(typenames))
    params = ["{0} {1}".format(arg[0], arg[1]) for arg in params]

    return_type = "auto"
    if name == "main":
        return_type = "int"

    funcdef = "{0}{1} {2}({3})".format(template, return_type, name,
                                       ", ".join(params))
    return funcdef + " {\n" + codegen_block(block, indent + 1) + "}\n\n"


def codegen_lambda(node):
    args = list(node.args)
    block = args.pop()
    assert isinstance(block, Block)
    params = ["auto {0}".format(param.name) for param in args]
    funcdef = "auto {0} = []({1})".format(node.name, ", ".join(params))
    return funcdef + " {\n" + codegen_block(block) + "\n};"



    # assert defnode.func.name == "def"
    # name = defnode.args[0]
    # cpp = io.StringIO()
    #
    # # template < typename T1, typename T2, typename T3 = decltype(5) >
    #
    #  # f"auto {name} ("
    # # cpp.write(f"std::shared_ptr<object> {name} (")
    # args = defnode.args[1:]
    # block = args.pop()
    # assert isinstance(block, Block)
    # template_preamble = ""
    # if len(args):
    #
    #     template_params = []
    #     for i, arg in enumerate(args, start=1):
    #         template_params.append(f"typename T{i} {arg.name}")
    #     template_preamble = "template <" + ", ".join(template_params) + ">"
    #     cpp_args = ", ".join([f" {a}" for a in args])
    #     cpp.write(cpp_args)

# unused
def _codegen_def_dynamic(defnode: Call):
    assert defnode.func.name == "def"
    name = defnode.args[0]
    cpp = io.StringIO()
    cpp.write(f"std::shared_ptr<object> {name} (")
    args = defnode.args[1:]
    block = args.pop()
    assert isinstance(block, Block)
    if len(args):
        cpp_args = ", ".join([f"std::shared_ptr<object> {a}" for a in args])
        cpp.write(cpp_args)
    cpp.write(") {\n")
    cpp.write(codegen_block(block))
    cpp.write("}")
    return cpp.getvalue()


def codegen(expr: Node):
    assert isinstance(expr, Module)
    s = codegen_node(expr)
    return cpp_preamble + s


def decltype(node):
    """Create C++ decltype statement"""
    if is_list(node):
        return "std::vector<decltype({0})>".format(value_type(node))
    else:
        return "decltype({0})".format(value_type(node))


def is_list(node):
    """Check if a node was assigned as a list"""
    if isinstance(node, ListLiteral):
        return True
    # elif isinstance(node, Assign):
    #     return is_list(node.rhs)  # dunno about this one
    elif isinstance(node, Identifier):
        # var = node.scopes.find(node.id)
        found_node, defining_context = find_def(node)
        if isinstance(defining_context, Assign) and is_list(defining_context.rhs):
            return True
        return (hasattr(var, "assigned_from") and not
        isinstance(var.assigned_from, ast.FunctionDef) and
                is_list(var.assigned_from.value))
    else:
        return False



def value_expr(node):
    """
    Follow all assignments down the rabbit hole in order to find
    the value expression of a name.
    The boundary is set to the current scope.
    """
    # return ValueExpressionVisitor().visit(node)


def value_type(node):
    """
    Guess the value type of a node based on the manipulations or assignments
    in the current scope.
    Special case: If node is a container like a list the value type inside the
    list is returned not the list type itself.
    """

    if not isinstance(node, Node):
        return
    elif isinstance(node, (IntegerLiteral, StringLiteral)):
        return value_expr(node)
    elif isinstance(node, Identifier):
        if node.name == 'True' or node.name == 'False':
            # return CLikeTranspiler().visit(node)
            return "true" if node.name == 'True' else "false"  # XXX??

            # var = node.scopes.find(node.id)
            found_node, defining_context = find_def(node)
            # if isinstance(defining_context, Assign) and is_list(
            #         defining_context.rhs):

            # if defined_before(var, node):
            #     return node.id
            # else:
            # return self.visit(var.assigned_from.value)

            # this is wrong
            return str(found_node)
            # return value_expr(found_node)

    # class ValueTypeVisitor(ast.NodeVisitor):
    #     def visit_Num(self, node):
    #         return value_expr(node)
    #
    #     def visit_Str(self, node):
    #         return value_expr(node)
    #
    #     def visit_Name(self, node):
    #         if node.id == 'True' or node.id == 'False':
    #             return CLikeTranspiler().visit(node)
    #
    #         var = node.scopes.find(node.id)
    #         if defined_before(var, node):
    #             return node.id
    #         else:
    #             return self.visit(var.assigned_from.value)
    #
    #     def visit_Call(self, node):
    #         params = ",".join([self.visit(arg) for arg in node.args])
    #         return "{0}({1})".format(node.func.id, params)
    #
    #     def visit_Assign(self, node):
    #         if isinstance(node.value, ast.List):
    #             if len(node.value.elts) > 0:
    #                 val = node.value.elts[0]
    #                 return self.visit(val)
    #             else:
    #                 target = node.targets[0]
    #                 var = node.scopes.find(target.id)
    #                 first_added_value = var.calls[0].args[0]
    #                 return value_expr(first_added_value)
    #         else:
    #             return self.visit(node.value)


# class ValueExpressionVisitor(ast.NodeVisitor):
#     def visit_Num(self, node):
#         return str(node.n)
#
#     def visit_Str(self, node):
#         return node.s
#
#     def visit_Name(self, node):
#         var = node.scopes.find(node.id)
#         if isinstance(var.assigned_from, ast.For):
#             it = var.assigned_from.iter
#             return "std::declval<typename decltype({0})::value_type>()".format(
#                    self.visit(it))
#         elif isinstance(var.assigned_from, ast.FunctionDef):
#             return var.id
#         else:
#             return self.visit(var.assigned_from.value)
#
#     def visit_Call(self, node):
#         params = ",".join([self.visit(arg) for arg in node.args])
#         return "{0}({1})".format(node.func.id, params)
#
#     def visit_Assign(self, node):
#         return self.visit(node.value)
#
#     def visit_BinOp(self, node):
#         return "{0} {1} {2}".format(self.visit(node.left),
#                                     CLikeTranspiler().visit(node.op),
#                                     self.visit(node.right))
#



# class ValueTypeVisitor(ast.NodeVisitor):
#     def visit_Num(self, node):
#         return value_expr(node)
#
#     def visit_Str(self, node):
#         return value_expr(node)
#
#     def visit_Name(self, node):
#         if node.id == 'True' or node.id == 'False':
#             return CLikeTranspiler().visit(node)
#
#         var = node.scopes.find(node.id)
#         if defined_before(var, node):
#             return node.id
#         else:
#             return self.visit(var.assigned_from.value)
#
#     def visit_Call(self, node):
#         params = ",".join([self.visit(arg) for arg in node.args])
#         return "{0}({1})".format(node.func.id, params)
#
#     def visit_Assign(self, node):
#         if isinstance(node.value, ast.List):
#             if len(node.value.elts) > 0:
#                 val = node.value.elts[0]
#                 return self.visit(val)
#             else:
#                 target = node.targets[0]
#                 var = node.scopes.find(target.id)
#                 first_added_value = var.calls[0].args[0]
#                 return value_expr(first_added_value)
#         else:
#             return self.visit(node.value)




def codegen_node(node: Union[Node, Any], indent=0):
    cpp = io.StringIO()

    if isinstance(node, Module):
        for modarg in node.args:
            if modarg.func.name == "def":
                defcode = codegen_def(modarg, indent)
                cpp.write(defcode)
            else:
                print("probably should handle", modarg)
    elif isinstance(node, Call): # not at module level
        if isinstance(node.func, Identifier):
            if node.func.name == "if":
                cpp.write(codegen_if(node))
            elif node.func.name == "def":
                print("need to handle nested def")
            else:
                cpp.write(node.func.name + "(" + ", ".join(map(codegen_node, node.args)) + ")")
        else:
            print("need to handle indirect call")

    elif isinstance(node, (Identifier, IntegerLiteral)):
        cpp.write(str(node))
    # elif isinstance(node, UnOp):
        # if node.func == "return":  # TODO fix UnOp func should be an identifier (although UnOp return should be converted to ColonBinOp earlier - or removed from language)
        #     cpp.write("return")
    elif isinstance(node, BinOp):
        if isinstance(node, ColonBinOp):
            assert isinstance(node, SyntaxColonBinOp)  # sanity check type system isn't leaking
            if node.lhs.name == "return":
                cpp.write("return " + codegen_node(node.args[1]))
            else:
                assert False
        elif isinstance(node, Assign) and isinstance(node.lhs, Identifier):
            found_def = find_def(node.lhs)

            if isinstance(node.rhs, ListLiteral) and not node.rhs.args:
                # need to 'infer' (via decltype) list type of empty list
                found_use = find_use(node)
                if found_use is not None:
                    print("yo")

                    found_use_node, found_use_context = found_use

                    if isinstance(found_use_context, AttributeAccess) and found_use_context.lhs is found_use_node and isinstance(found_use_context.rhs, Call) and found_use_context.rhs.func.name == "append":
                        apnd = found_use_context.rhs
                        assert len(apnd.args) == 1
                        rhs_str = "std::vector<decltype({})>{{}}".format(codegen_node(apnd.args[0]))
                    else:
                        raise CodeGenError("list error, dunno what to do with this:", node)
                else:
                    raise CodeGenError("Unused empty list in template codegen", node)
            else:
                rhs_str = codegen_node(node.rhs)

            assign_str = " ".join([codegen_node(node.lhs), node.func, rhs_str])

            if found_def is None:
                assign_str = "auto " + assign_str
            cpp.write(assign_str)
        else:
            binop_str = None

            separator = " "
            if isinstance(node, AttributeAccess):
                separator = ""

                if isinstance(node.rhs, Call) and node.rhs.func.name == "append":
                    apnd = node.rhs
                    assert len(apnd.args) == 1

                    if isinstance(node.lhs, ListLiteral) or ((found_def := find_def(node.lhs)) is not None and isinstance(found_def[1], Assign) and isinstance(found_def[1].rhs, ListLiteral)):
                        binop_str = "{}.push_back({})".format(codegen_node(node.lhs), codegen_node(apnd.args[0]))

            if binop_str is None:
                cpp.write(separator.join([codegen_node(node.lhs), node.func, codegen_node(node.rhs)]))
            else:
                cpp.write(binop_str)

    elif isinstance(node, ListLiteral):
        if node.args:
            elements = [codegen_node(e) for e in node.args]
                # value_type = decltype(node.elts[0])
            return "std::vector<decltype({})>{{{}}}".format(elements[0], ", ".join(elements))
        else:
            assert False

                # "std::vector<{0}>{{{1}}}""
            #
            # "std::vector<decltype({0})>".format(value_type(node))

            # return "std::vector<{0}>{{{1}}}".format(value_type,
            #                                         ", ".join(elements))

            raise CodeGenError("Cannot create vector without elements (in template generation mode)")


    return cpp.getvalue()
