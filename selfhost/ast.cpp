
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

//#include <concepts>
//#include <ranges>
//#include <numeric>


#include "ceto.h"


#include <map>
#include <typeinfo>
#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
;
namespace py = pybind11;
    template <typename T1, typename T2>
auto join(const T1& v, const T2& to_string, const decltype(std::string {""})&  sep = std::string {""}) -> auto {
if (ceto::mado(v)->empty()) {
            return std::string {""};
        }
        return std::accumulate(ceto::mado(v)->cbegin() + 1, ceto::mado(v)->cend(), to_string(ceto::maybe_bounds_check_access(v,0)), [&to_string, &sep](const auto &a, const auto &el) {
                if constexpr (!std::is_void_v<decltype(((a + sep) + to_string(el)))>) { return ((a + sep) + to_string(el)); } else { static_cast<void>(((a + sep) + to_string(el))); };
                });
    }

struct Node : ceto::shared_object {

    std::shared_ptr<const Node> func;

    std::vector<std::shared_ptr<const Node>> args;

    py::tuple source;

    py::object parent = py::none(); static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(py::none()), std::remove_cvref_t<decltype(parent)>>);

    std::shared_ptr<const Node> declared_type = nullptr; static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(nullptr), std::remove_cvref_t<decltype(declared_type)>>);

    py::object scope = py::none(); static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(py::none()), std::remove_cvref_t<decltype(scope)>>);

         virtual inline auto repr() const -> std::string {
            const std::string classname = ceto::mado(typeid((*this)))->name(); static_assert(ceto::is_non_aggregate_init_and_if_convertible_then_non_narrowing_v<decltype(ceto::mado(typeid((*this)))->name()), std::remove_cvref_t<decltype(classname)>>);
            const auto csv = join(this -> args, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {", "});
            return (((((classname + std::string {"("}) + ceto::mado(this -> func)->repr()) + std::string {")(["}) + csv) + std::string {"])"});
        }

         virtual inline auto name() const -> std::optional<std::string> {
            return std::nullopt;
        }

    explicit Node(const std::shared_ptr<const Node>&  func, const std::vector<std::shared_ptr<const Node>>&  args, const decltype(py::tuple{}) source = py::tuple{}) : func(func), args(args), source(source) {
    }

    Node() = delete;

};

struct UnOp : public Node {

};

struct LeftAssociativeUnOp : public Node {

};

struct BinOp : public Node {

        inline auto lhs() const -> auto {
            return ceto::maybe_bounds_check_access(this -> args,0);
        }

        inline auto rhs() const -> auto {
            return ceto::maybe_bounds_check_access(this -> args,1);
        }

        inline auto repr() const -> std::string {
            return join(std::vector {{this -> lhs(), this -> func, this -> rhs()}}, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {" "});
        }

};

struct TypeOp : public BinOp {

};

struct SyntaxTypeOp : public TypeOp {

};

struct AttributeAccess : public BinOp {

        inline auto repr() const -> std::string {
            return ((ceto::mado(this -> lhs())->repr() + std::string {"."}) + ceto::mado(this -> rhs())->repr());
        }

};

struct ArrowOp : public BinOp {

};

struct ScopeResolution : public BinOp {

};

struct Assign : public BinOp {

};

struct Identifier : public std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> {

    std::string _name;

        inline auto repr() const -> std::string {
            return (this -> _name);
        }

        inline auto name() const -> std::optional<std::string> {
            return (this -> _name);
        }

    explicit Identifier(const std::string&  name, const decltype(py::tuple{}) source = py::tuple{}) : std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> (nullptr, std::vector<std::shared_ptr<const Node>>{}, source), _name(name) {
    }

    Identifier() = delete;

};

struct Call : public Node {

        inline auto repr() const -> std::string {
            const auto csv = join(this -> args, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {", "});
            return (((ceto::mado(this -> func)->repr() + std::string {"("}) + csv) + std::string {")"});
        }

};

struct ArrayAccess : public Node {

        inline auto repr() const -> std::string {
            const auto csv = join(this -> args, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {", "});
            return (((ceto::mado(this -> func)->repr() + std::string {"["}) + csv) + std::string {"]"});
        }

};

struct BracedCall : public Node {

        inline auto repr() const -> std::string {
            const auto csv = join(this -> args, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {", "});
            return (((ceto::mado(this -> func)->repr() + std::string {"{"}) + csv) + std::string {"}"});
        }

};

struct Template : public Node {

        inline auto repr() const -> std::string {
            const auto csv = join(this -> args, [](const auto &a) {
                    if constexpr (!std::is_void_v<decltype(ceto::mado(a)->repr())>) { return ceto::mado(a)->repr(); } else { static_cast<void>(ceto::mado(a)->repr()); };
                    }, std::string {", "});
            return (((ceto::mado(this -> func)->repr() + std::string {"<"}) + csv) + std::string {">"});
        }

};

struct IntegerLiteral : public std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> {

    py::object integer;

        inline auto repr() const -> std::string {
            return std::string(py::str(this -> integer));
        }

    explicit IntegerLiteral(const py::object  integer, const decltype(py::tuple{}) source = py::tuple{}) : std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> (nullptr, std::vector<std::shared_ptr<const Node>>{}, source), integer(integer) {
    }

    IntegerLiteral() = delete;

};

    inline auto string_replace(const std::string&  source, const std::string&  from, const std::string&  to) -> auto {
        auto new_string { std::string() } ;
        ceto::mado(new_string)->reserve(ceto::mado(source)->length());
        std::string::size_type last_pos { 0 } ; static_assert(std::is_convertible_v<decltype(0), decltype(last_pos)>);
        std::string::size_type find_pos { 0 } ; static_assert(std::is_convertible_v<decltype(0), decltype(find_pos)>);
while (std::string::npos != (find_pos = ceto::mado(source)->find(from, last_pos))) {            ceto::mado(new_string)->append(source, last_pos, find_pos - last_pos);
            new_string += to;
            last_pos = (find_pos + ceto::mado(from)->length());
        }
        ceto::mado(new_string)->append(source, last_pos, ceto::mado(source)->length() - last_pos);
        return new_string;
    }

struct StringLiteral : public std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> {

    std::string str;

    std::shared_ptr<const Identifier> prefix;

    std::shared_ptr<const Identifier> suffix;

        inline auto escaped() const -> auto {
            auto s { string_replace(this -> str, std::string {"\n"}, std::string {"\\n"}) } ;
            s = string_replace(s, std::string {"\""}, std::string {"\""});
            s = ((std::string {"\""} + s) + std::string {"\""});
            return s;
        }

        inline auto repr() const -> std::string {
            auto s { this -> escaped() } ;
if (this -> prefix) {
                s = (ceto::mad(ceto::mado(this -> prefix)->name())->value() + s);
            }
if (this -> suffix) {
                s += ceto::mad(ceto::mado(this -> suffix)->name())->value();
            }
            return s;
        }

    explicit StringLiteral(const std::string&  str, const std::shared_ptr<const Identifier>&  prefix, const std::shared_ptr<const Identifier>&  suffix, const decltype(py::tuple{}) source = py::tuple{}) : std::type_identity_t<decltype(Node(nullptr, std::vector<std::shared_ptr<const Node>>{}, std::declval<std::remove_cvref_t<const decltype(py::tuple{})>>()))> (nullptr, std::vector<std::shared_ptr<const Node>>{}, source), str(str), prefix(prefix), suffix(suffix) {
    }

    StringLiteral() = delete;

};

    inline auto example_macro_body_workaround_no_fptr_syntax_yet(const std::map<std::string,std::shared_ptr<const Node>>  matches) -> std::shared_ptr<const Node> {
        return nullptr;
    }

    inline auto macro_trampoline(const uintptr_t  fptr, const std::map<std::string,std::shared_ptr<const Node>>  matches) -> auto {
        const auto f = reinterpret_cast<decltype(&example_macro_body_workaround_no_fptr_syntax_yet)>(fptr);
        return (*f)(matches);
    }

PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<const Node>>);
PYBIND11_MAKE_OPAQUE(std::map<std::string,std::shared_ptr<const Node>>);

PYBIND11_MODULE(_abstractsyntaxtree, m) {
;
[]( auto &&  m) {
        py::bind_vector<std::vector<std::shared_ptr<const Node>>>(m, "NodeVector");
        py::bind_map<std::map<std::string,std::shared_ptr<const Node>>>(m, "StringNodeMap");
        auto node { ceto::mado(ceto::mado(ceto::mado(ceto::mado(ceto::mado(ceto::mado(ceto::mado(ceto::mado(py::class_<Node,std::shared_ptr<Node>>(m, "Node"))->def_readwrite("func", (&Node::func)))->def_readwrite("args", (&Node::args)))->def_readwrite("parent", (&Node::parent)))->def_readwrite("declared_type", (&Node::declared_type)))->def_readwrite("scope", (&Node::scope)))->def_readwrite("source", (&Node::source)))->def("__repr__", (&Node::repr)))->def("name", (&Node::name)) } ;
        ceto::mado(py::class_<Identifier,std::shared_ptr<Identifier>>(m, "Identifier", node))->def(py::init<const std::string &,py::tuple>());
        ceto::mado(m)->def("macro_trampoline", (&macro_trampoline), "macro trampoline");
        return;
        }(m);
};
