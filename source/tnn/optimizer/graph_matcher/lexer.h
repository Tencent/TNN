// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.


#ifndef TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LEXER_H_
#define TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LEXER_H_

#include <string>
#include <sstream>
#include <memory>
#include <ostream>
#include <assert.h>
#include <mutex>
#include <set>

#include "tnn/core/macro.h"
#include "tnn/optimizer/graph_matcher/ir.h"
#include "tnn/optimizer/graph_matcher/graph_registry.h"
#include "tnn/optimizer/graph_matcher/logger.h"

namespace TNN_NS {

#define TNN_ALL_TOKEN_KINDS(_)                   \
  _(TK_EOF, "eof", "")                           \
  _(TK_WHITESPACE, "whitespace", "")             \
  _(TK_WHITESPACE_EOF, "whitespace_eof", "")     \
  _(TK_NUMBER, "number", "")                     \
  _(TK_NEWLINE, "newline", "")                   \
  _(TK_IDENT, "identifier", "")                  \
  _(TK_LAYER_TYPE, "layer_type", "")             \
  _(TK_GRAPH, "graph", "graph")                       \
  _(TK_RETURN, "return", "return")                     \
  _(TK_GRAPH_FUNCTION, "graph_function", "")            \

static const char* valid_single_char_tokens = "+<>#@{}()[]=%:,";

std::string layerTypeName(LayerType type);

enum TokenKind {
  TK_DUMMY_START = 256,
#define DEFINE_TOKEN(tok, _, _2) tok,
  TNN_ALL_TOKEN_KINDS(DEFINE_TOKEN)
#undef DEFINE_TOKEN
};

std::string tokenName(int kind);

struct SubStr {
/* TODO :
    add checking macro  
    check if offset and len are in range in all functions
 */
public:
    SubStr(std::string str) {
        new (this) SubStr(str, 0, str.length());
    }

    SubStr(std::string str, int of, int l) {
        sptr_ = std::make_shared<std::string>(str);
        offset_ = of;
        len_ = l;
    }

    SubStr(const SubStr &str, int of, int l=-1) :
        sptr_(str.sptr_), offset_(of+str.offset_), len_(l) {
        if (len_ == -1) {
            len_ = str.len() - of;
        }
    }

    SubStr():sptr_(nullptr), offset_(0), len_(0) {}

    void shift(size_t n) {offset_ += n; len_ -= n;}

    std::string str() const {
        if (sptr_ &&
            offset_ >= 0 && 
            offset_ < sptr_->length() &&
            len_ >= 0 && 
            offset_ + len_ <= sptr_->length()) 
        {
            return sptr_->substr(offset_, len_);
        }
        return std::string();
    }

    void highlight(std::ostream& out) const {
        const std::string& str = *sptr_;
        int begin = offset();
        int end = offset();
        while (begin > 0 && str[begin - 1] != '\n') --begin;
        while (end < str.size() && str[end] != '\n') ++end;
        out << str.substr(0, end) << "\n";
        out << std::string(offset() - begin, ' ');
        int len = std::min(this->len(), end - offset());
        out << std::string(len, '~')
            << (len < this->len() ? "...  <--- HERE" : " <--- HERE");
        out << str.substr(end);
        if (str.size() > 0 && str.back() != '\n')
        out << "\n";
    }

    std::string full() const {
        return *sptr_;
    }

    char &at(size_t pos=0) const {
        return (*sptr_).at(offset_ + pos);
    }

    char &operator[](size_t pos) const {
        return this->at(pos);
    }

    size_t capacity() const {
        return sptr_->length() - size_t(offset_);
    }

    int len() const {
        return len_;
    }

    int offset() const {
        return offset_;
    }

    bool operator<(const SubStr &o) const {
        if (sptr_ == o.sptr_) {
            return offset_ < o.offset_ || (offset_ == o.offset_ && len_ < o.len_);
        }
        return size_t(sptr_.get()) < size_t(o.sptr_.get());
    }

private:
    std::shared_ptr<std::string> sptr_;
    int offset_ = 0;
    int len_ = 0;
};


struct TrieTree;
using TrieTreePtr = std::unique_ptr<TrieTree>;
struct TrieTree {
    TrieTree() : kind(0){};

    void insert(const char * str, int token_kind)  {
        if (!str) return;
        if (str[0] == '\0') {
            assert(kind == 0);
            kind = token_kind;
            return;
        }
        for(size_t i=0;i<nexts.size();i++) {
            if (next_chars[i] == str[0]) {
                nexts[i]->insert(str+1, token_kind);
                return;
            }
        }

        next_chars.emplace_back(str[0]);
        nexts.emplace_back(std::unique_ptr<TrieTree>(new TrieTree()));
        nexts.back()->insert(str+1, token_kind);
    }

    void remove(const char * str)  {
        if (!str) return;
        if (str[0] == '\0') {
            kind = 0;
            return;
        }
        for(size_t i=0;i<nexts.size();i++) {
            if (next_chars[i] == str[0]) {
                nexts[i]->remove(str+1);
                return;
            }
        }
    }

    int kind = 0;
    std::vector<TrieTreePtr> nexts;
    std::vector<char> next_chars;

};

struct Token {
    Token() :kind(0) {};
    Token(int _kind, SubStr _str) : kind(_kind), str(_str) {}
    int kind;
    SubStr str;

    std::string text() const {
        return str.str();
    }

    std::string name() const {
        return tokenName(kind);
    }

    bool operator<(const Token &o) const {
        return kind < o.kind ||(kind == o.kind && str < o.str);
    }
};

struct Tokenizer {
    Tokenizer() : head(new TrieTree()) {
        for (const char* c = valid_single_char_tokens; *c!='\0'; c++) {
            std::string str(1, *c);
            head->insert(str.c_str(), *c);
        }

#define ADD_TOKEN(tok, _, tokstring)   \
if (*(tokstring) != '\0') {         \
    head->insert((tokstring), (tok)); \
}
    TNN_ALL_TOKEN_KINDS(ADD_TOKEN)
#undef ADD_TOKEN

        for (auto &pair : GetGlobalLayerTypeMap()) {
            auto &str = pair.first;
            LayerType type = pair.second;
            head->insert(str.c_str(), TK_LAYER_TYPE);
        }

    }

    Tokenizer& operator=(Tokenizer && rhs) {
        head.swap(rhs.head);
        if (registry) {
            RAISE_ON_ERROR(registry->unRegisterTokenizer(this));
        }
        if (rhs.registry) {
            RAISE_ON_ERROR(rhs.registry->unRegisterTokenizer(&rhs));
        }
        registry = rhs.registry;
        rhs.registry = nullptr;
        if (registry) {
            RAISE_ON_ERROR(registry->registerTokenizer(this));
        }
        return *this;
    }
    
    ~Tokenizer()  {
        if (registry) {
            registry->unRegisterTokenizer(this);
        } 
    }

    void bindGraphRegistry(GraphRegistry * _registry) {
        if (_registry) {
            RAISE_ON_ERROR(_registry->registerTokenizer(this));
            registry = _registry;
        }
    }

    void onNewToken(std::string str, int kind) {
        // DEBUG("Tokenizer %p got graph %s kind:%d", this, str.c_str(), kind);
        head->insert(str.c_str(), kind);
    }

    void onDelete(std::string str) {
        // DEBUG("Tokenizer %p remove graph %s", this, str.c_str());
        head->remove(str.c_str());
    }

    bool isNumber(SubStr& str, Token * token) {
        int cnt = 0;
        while(cnt < str.capacity() && '0' <= str[cnt] && str[cnt] <= '9') {
            cnt ++;
        }
        if (cnt == 0) {
            return false;
        }
        *token = Token(TK_NUMBER, SubStr(str, 0, cnt));
        return true;
    }

    bool match(SubStr &str, bool whitespace_token, Token * tok) {
        if (str.capacity() == 0) {
            *tok = Token(TK_EOF, SubStr(str, 0));
            return true;
        }

        size_t cnt = 0;
        while(cnt < str.capacity() && str[cnt] == ' ')  {
            cnt ++;
        }

        if (whitespace_token && cnt > 0) {
            *tok = Token(TK_WHITESPACE, SubStr(str, 0, cnt));
            return true;
        }

        str.shift(cnt);

        if (isNumber(str, tok)) {
            return true;
        }

        if (str[0] == '\n') {
            *tok = Token(TK_NEWLINE, SubStr(str, 0, 1));
            return true;
        }

        if (str.capacity() == 0) {
            *tok = Token(TK_WHITESPACE_EOF, SubStr(str, 0));
            return true;
        }

        auto validIdentifier = [](size_t i, char &c) {
            return isalpha(c) || c == '_' || (i > 0 && isdigit(c));
        };
        

        bool matched = false;
        bool ident = true;
        TrieTree * cur = head.get();
        for (size_t i = 0; i < str.capacity() && (ident || cur != nullptr); i++) {
            ident = ident && validIdentifier(i, str[i]);
            if (ident) {
                matched = true;
                *tok = Token(TK_IDENT, SubStr(str, 0, i+1));
            }

            if (cur) {
                size_t next_pos = 0;
                for (size_t e = cur->nexts.size(); next_pos < e;
                    ++next_pos) {
                    if (cur->next_chars[next_pos] == str[i])
                        break;
                }

                cur = (next_pos == cur->nexts.size())
                    ? nullptr
                    : cur->nexts[next_pos].get();

                if (cur && cur->kind != 0) {
                    matched = true;
                    *tok = Token(cur->kind, SubStr(str, 0, i+1));
                }
            }
        }
        return matched;
    }


private:
    TrieTreePtr head;
    GraphRegistry * registry = nullptr;
};


void expect(const Token &tk, const int kind);
void expect(const Token &tk, const std::vector<int> kind);

void unexpect(const Token &tk);

void reportError(const std::string &msg, const Token &tok);

struct Lexer {
    Lexer(SubStr str): source_(str) { step();};
    Lexer(SubStr str, GraphRegistry* registry): source_(str) { t_.bindGraphRegistry(registry);  step();};

    Lexer(const Lexer &)=delete;
    Lexer(const Lexer &&)=delete;
    Lexer& operator=(Lexer && rhs) {
        t_ = std::move(rhs.t_);
        source_ = std::move(rhs.source_);
        prev_ = std::move(rhs.prev_);
        next_tokens = std::move(rhs.next_tokens);
        return *this;
    }

    Token& lookahead() {
        if (next_tokens.size() < 2) {
            step();
        }
        return next_tokens[1];
    }

    const Token& cur() const {
        if (next_tokens.size() == 0) 
            throw std::runtime_error("Lexer got empty token.");
        return next_tokens.front();
    }

    const Token &prev() const {
        return prev_;
    }

    Token next() {
        Token t = cur();
        prev_ = t;
        next_tokens.erase(next_tokens.begin());
        if (next_tokens.size() == 0) {
            step();
        }
        return t;
    }

    void errorHere() {
        std::stringstream ss;
        SubStr error_pos(source_, 0, 1);
        ss << "got invalid token :\n";
        error_pos.highlight(ss);
        throw std::runtime_error(ss.str());
    }

    void step() {
        Token t = parse();
        next_tokens.emplace_back(t);
    }

    Token parse() {
        Token tok;
        if (!t_.match(source_, true, &tok)) {
            errorHere(); 
        }
        source_.shift(tok.text().length());
        return tok;
    }

private: 
    Tokenizer t_;
    SubStr source_;
    Token prev_;

    std::vector<Token> next_tokens;

};

}


#endif // TNN_SOURCE_TNN_NET_OPTIMIZER_GRAPH_MATCHER_LEXER_H_
