# Python 3.7.7~3.10.0

- [Python 3.7.7~3.10.0](#python-3773100)
  - [記法](#記法)
    - [型アノテーション　注釈](#型アノテーション注釈)
    - [while](#while)
    - [lambda](#lambda)
    - [文字列　 f 文字,%フォーマット指定子](#文字列-f-文字フォーマット指定子)
    - [アンダースコアの使い方](#アンダースコアの使い方)
    - [辞書](#辞書)
    - [デコレータ](#デコレータ)
    - [セイウチ](#セイウチ)
    - [可変長引数、スター](#可変長引数スター)
    - [...](#)
    - [構造的パターンマッチ(3.10~)](#構造的パターンマッチ310)
    - [例外](#例外)
  - [組み込み関数](#組み込み関数)
    - [XOR](#xor)
    - [スライス](#スライス)
    - [set](#set)
    - [isinstance](#isinstance)
    - [リストの演算](#リストの演算)
    - [split](#split)
    - [文字変換置換](#文字変換置換)
    - [format 関数](#format-関数)
    - [just](#just)
    - [round](#round)
    - [要素の真偽判定](#要素の真偽判定)
    - [イテレータ](#イテレータ)
  - [標準ライブラリ](#標準ライブラリ)
    - [pathlib](#pathlib)
    - [re : 正規表現](#re--正規表現)
    - [enum](#enum)
    - [dataclass](#dataclass)
    - [decimal](#decimal)
    - [logger](#logger)
    - [argparse](#argparse)
    - [unittest (+pytest)](#unittest-pytest)
    - [pytest コンソールコマンド](#pytest-コンソールコマンド)
    - [Radon コマンド](#radon-コマンド)
    - [py_compile](#py_compile)
    - [subprocess](#subprocess)
    - [ctype](#ctype)
    - [抽象クラス(abc)](#抽象クラスabc)
    - [threading](#threading)
    - [concurrent](#concurrent)
    - [socket](#socket)
    - [tk](#tk)
- [Python launcher](#python-launcher)
- [pip](#pip)

---

---

## 記法

### 型アノテーション　注釈

```python
def note(memo: str) -> str:
    return "Note: " + memo
```

-   `memo:str`
    引数`memo`が、`str`型であることを注釈
-   `->`
    関数の返り値が `str` 型であることの注釈

```python
#v3.9~
list[int,int]
dict[str:bool]
```

### while

```python{cmd=true}
i=0

while i<5:
    print(i)
    i+=1
else:
    print("END")
```

-   `continue`その後の処理をスキップ
-   `break`途中で終了
-   `else`正常処理後の処理、ループ条件式が`False`になるまで実行されたら発動

### lambda

```python
def func(x,y,z):
    return x*y*z

func=lambda x,y,z:x*y*z
```

-   三項演算子　`[True時処理1] if [条件1] else [True時処理2] if [条件2] else [False時処理]`

```python{cmd=true}
scores = [52, 85, 90, 40, 80, 30, 5]
results = map(lambda x: 'PASS' if x >= 80 else "DEATH" if x <= 30 else "FAIL", scores)

print(list(results))
```

-   OR `[Trueと評価される値] or [Falseの場合の値]`

```python{cmd=true}
fruits = ['Apple', '', 'Orange']
results = map(lambda x: x or "(n/a)", fruits)

print(list(results))
```

### 文字列　 f 文字,%フォーマット指定子

```python{cmd=true}
a="Snark"
b="boojum"

print(f"{a} was {b}")
print("%s was %s!" %(a,b))

li=["LONDON","BERLIN","PARIS"]

print("Hello %s was fallen" %(li,))
```

### アンダースコアの使い方

-   Return 値を無視する

```python{cmd=true}
def date():
    month = 7
    day = 14
    return month, day

print(date())
m, d = date()
print(f"{m}gatsu {d}nichi")
_, d = date()
print(d)
```

-   オブジェクトの命名

```python{cmd=true}
class Under_Score():
  aa = "クラス変数"
  _bb = "プライベート"  # アクセス可能
  __cc = "NameMangling"  # アクセス一部可能
  def_ = "予約語を使いたいとき"
  __magicmethod__="新しく定義するのはやめよう"

US = Under_Score()
Obj = set(["aa", "_bb", "__cc", "def_", "__magicmethod__"]) & set(dir(US))
#クラス変数一覧
print(Obj)

## "__cc"はクラス変数にないので
print(US._Under_Score__cc)
```

-   数字を見やすく

```python{cmd=true}
print(100000000000)
print(100_000_000_000)
```

-   Python は snake_case?

### 辞書

-   作成

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}
print(di)

di = dict(Tokyo=546, Osaka=122, Fukuoka=35, Aichi=45, Okinawa=105)
print(di)

di = [("Tokyo", 546), ("Osaka", 122), ("Fukuoka", 35),
      ("Aichi", 45), ("Okinawa", 105)]
print(dict(di))

format_dict = dict.fromkeys(
    ["Tokyo", "Osaka", "Fukuoka", "Aichi", "Okinawa"], 0
)
print(format_dict)
```

-   取得

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}

print(di["Tokyo"])
# print(di["Akita"]) keyがないとKeyError

print(di.get("Tokyo"))
print(di.get("Akita"))
```

-   要素

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}

print(di.keys())
print(di.values())
print(di.items())

print(list(di.keys()))
```

-   追加

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}

di.setdefault("Hokkaido",59)
# 上書きされない
di.setdefault("Okinawa",159)
print(di)

di["Kyoto"]=13
# 上書きされる
di["Okinawa"]=159
print(di)

di.update({"Tokyo":3000,"Kanazawa":12})
print(di)
```

-   削除

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}

# 最後の要素を除く
ram=di.popitem()
print(di)
print(ram)

del di["Osaka"], di["Aichi"]
print(di)

di.clear()
print(di)
```

-   ソート

```python{cmd=true}
di = {
    "Tokyo": 546, "Osaka": 122, "Fukuoka": 35, "Aichi": 45, "Okinawa": 105
}


so=sorted(di.items(),key=lambda x:x[1],reverse=True)
print(so)

so=sorted(di,key=len)
print(so)
```

### デコレータ

```python{cmd=true}
import functools
import time

class MainMethod():
    def deco(method):  # noqa
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            print("¥n*****")
            method(self, *args, **kwargs)
            print("*****¥n")
        return wrapper

    @deco
    def print_console(self, time_waits):
        print("Hello mad pianist")
        for i in range(time_waits + 1):
            time.sleep(1)
            print(f"¥rTimeCount {i}:", end="")
        else:
            print("Stopping as you wish")


if __name__ == "__main__":
    MM = MainMethod()
    MM.print_console(5)
```

### セイウチ

```python{cmd=true}
i = 0
ii = [(i := 2 + i) for _ in range(20)]

print(ii)
```

### 可変長引数、スター

```python{cmd=true}
import math


def nature(arg1: int, arg2: int, arg3: int):
    recfac1 = 1 / math.factorial(arg1)
    recfac2 = 1 / math.factorial(arg2)
    recfac3 = 1 / math.factorial(arg3)

    return recfac1 + recfac2 + recfac3

args = 1, 2, 3
print(nature(*args))


def nature(*args: int):
    recfac = 0
    for arg in args:
        recfac += 1 / math.factorial(arg)

    return recfac


print(nature(*[i for i in range(20)]))

def kwarg(**kwargs):
    for kw in kwargs.items():
        print(kw)

dd = {
    "A": 1,
    "B": 2,
    "C": 3
}
kwarg(**dd)
```

### ...

```python{cmd=true}
ss=...
print(ss)
print(str(type(ss)))
```

### 構造的パターンマッチ(3.10~)

```python{cmd=true}
def matching(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case _:
            return "Something's wrong with the Internet"

print(matching(400))
```

### 例外

```python
class YourOwnException(Exception):
    def __init__(self, arg=""):
        pass

    def __str__(self):
        return "MESSAGE"

raise YourOWnException("ARGS")
```

---

## 組み込み関数

### XOR

```python{cmd=true}
res=0^1
print("0^1=",res)
res=1^1
print("1^1=",res)
```

### スライス

文字列もリストも同じ

```python{cmd=true}
fib=[0,1,1,2,3,5,8,13,21,34,55,89,144,233,377,610]

print(fib[4])
print(fib[1:6])
print(fib[-3:-1])
print(fib[7:-1:2])
```

### set

```python{cmd=true}
li=[5,7,3,9,1,0,2,8,4,6]
print(type(set(li)),set(li)) ## CodeChunkでTypeがタイプされない
```

### isinstance

第一引数が第二引数のオブジェクトなら True を返す

```python{cmd=true}
res=isinstance("strings",str)
print(res)

# 整数かチェック
print(0.1.is_integer())
print(1.0.is_integer())
```

### リストの演算

```python{cmd=true}
li=["AA","SS","DD","FF","GG"]
st=["EE","SS","ZZ"]

res=set(li)&set(st)

print(res)

res=set(li)|set(st)

print(res)

res=set(li)^set(st)

print(list(res))
```

```python{cmd=true}
li=["AA","SS","DD","FF","GG"]
st=["EE","SS","ZZ"]

li.extend(st)
print("extend")
print(li)

li=["AA","SS","DD","FF","GG"]
st=["EE","SS","ZZ"]

li.append(st)
print("append")
print(li)
```

-   `&`AND
-   `|`OR
-   `^`XOR

### split

```python{cmd=true}
st="17:55:37:159"
res=st.split(":")

print(res)

res=st.split(":",2)

print(res)
```

### 文字変換置換

-   辞書で変換

```python{cmd=true}
# 1文字限定らしい
d = {
    " ": "*",
    "a": "3",
    "i": "1"
}
tbl = str.maketrans(d)

cont = "Laid liar lying foo"
print(cont.translate(tbl))
```

```python{cmd=true}
cont="stIllNEss In tImE"

#大文字に
print("[upper]",cont.upper())
#小文字に
print("[lower]",cont.lower())
#先頭大文字
print("[capitalize]",cont.capitalize())
#先頭大文字
print("[title]",cont.title())
#大小反転
print("[swapcase]",cont.swapcase())
```

-   str, repr, ascii

```python{cmd=true}
ii = "¥nmobilis in mobli¥n"

print(str(ii))
print(repr(ii))
print(ascii(ii))
```

### format 関数

{}.format(A) -> A
`{2:<20,.3f}`{インデックス番号:文字寄せ 最小幅 カンマ区切り 小数点以下の桁数 変数型}

-   変数型の指定
    | 記号 | type | 注釈 |
    | :--: | :--------------------: | :--------------------------: |
    | `s` | 文字列 | デフォルト　省略可 |
    | `d` | 整数(10 進数) | bin は`b` hex は`x` |
    | `e` | 浮動小数点(指数表記) | `E`だと E が使われる |
    | `f` | 浮動小数点(小数点表記) | `F`の場合、`NAN`/`INF`と表示 |

-   3.6 以降
    `f"{変数:.3f}`

```python{cmd=true}
st = 'QQ{:4}QQ'.format('A')
print(st)
st = 'QQ{:<10}QQ'.format('AA')
print(st)
st = 'QQ{:>10}QQ'.format('AA')
print(st)
st = '[{:^10}]'.format('AA')
print(st)
st = '[{:0^10}]'.format('AAA')
print(st)

nume = 123456789
print(f'{nume:,}')
print(f'{nume:.0e}')
nume = 1.23456789
print(f'{nume:.2f}')
nume = 1
print(f'{nume:%}')
print(f'{nume:.2%}')
a = -100
b = 100
print(f'{a:+} {b:+}')

import datetime as dt
today = dt.datetime.now()

print("{0:%YYear%mMonth%dDay---%IHour%MMin%SSec}".format(today))


# 反転
st="Color"
ts=st[::-1]
print(ts)
```

### just

```python{cmd=true}
st="ABC"
print(st.ljust(10,"_"))
print(st.rjust(10,"_"))
print(st.center(10,"_"))
```

### round

```python{cmd=true}
st=round(1.2345)
print(st)

# 偶数を選ぶので四捨五入ではない
st=round(1.5)
print(st)
st=round(2.5)
print(st)
```

### 要素の真偽判定

-   **all** 全ての要素が真の時、True を返す。
-   **any** いずれかの要素が真の時、True を返す。

```python{cmd=true}
case_ = all([True, True, True])
print(case_)

case_ = all([True, False, True])
print(case_)

case_ = any([False, False, True])
print(case_)

case_ = all([False, False, False])
print(case_)
```

### イテレータ

-   range

```python
for i in range(0, 10, 2):
    print(i)
```

-   enumerate

```python
iter = ["A", "A", "A", "A", "A"]
for index, item in enumerate(iter, start=3):
    print(f"{index}:{item}")
```

-   zip

```python{cmd=true}
iterA = ["A", "A", "A", "A", "A"]
iterB = ["B", "B", "B", "B"]
for index, item in enumerate(zip(iterA, iterB)):
    print(f"{index}:{item}")
```

---

## 標準ライブラリ

### pathlib

-   インポート

```python{cmd=true}
from pathlib import Path

pt = Path("path/to/path")
print(pt)
```

-   判定

```python
pt = Path("path/to/file")
# ファイル
print(pt.is_file())
# ディレクトリ
print(pt.is_dir())
# 存在
print(pt.exists())
```

-   連結

```python
pt = Path("path/to/folder")
dr = Path("dir/to/file")

com = Path(pt, dr)
com = pt / dr
```

-   一覧取得

```python
# 全て取得、ワイルドカードで指定できる
list(path_obj.glob("*"))
```

-   相対パスを絶対パスに変換

```python
path_obj.resolve()
print(path_obj.is_absolute())
```

-   作業ディレクトリ CurrentWorkingDirectory

```python
path_obj.cwd()
```

-   その他の操作

```python
# ファイル名
path_obj.name
# 拡張子なしの名前
path_obj.stem
# 拡張子
path_obj.suffix
# 親ディレクトリ
path_obj.parent
path_obj.parents[]
# Pathオブジェクトを文字列で欲しいときはstrに変換
str(path_obj)
```

### re : 正規表現

-   `re.match(pattern,content)`content の先頭が pattern に一致するか調べる
-   `re.search(pattern,content)`先頭以外も、最初にマッチしたものを返す
-   `re.findall(pattern,content)` 全て

```python{cmd=true}
import re

content = r'What is the longest sentence in the world?'
pattern = 'Wha'

result = re.match(pattern, content)

if result: #none以外の場合
  print(result)
  print(result.span())
  print(result.group())

li=[
  "SAPPORO",
  "MORIOKA",
  "SENDAI",
  "UTSUNOMIYA",
  "MAEBASHI",
  "YOKOHAMA",
  "KANAZAWA",
  "NAGOYA"
]

ss="MA"
res=[
  i for i in li if re.search(ss,i)
]

print(f"MATCH-> {ss}",res)
```

-   `r"`はエスケープ文字を無視する。raw 生の

```python{cmd=true}
strings="\\\nHello, The World\n"
print(strings)

strings=r"\\\nHello, The World\n"
print(strings)
```

|  文字   |        説明         |     同様      |    例    |        マッチする        | マッチしない |
| :-----: | :-----------------: | :-----------: | :------: | :----------------------: | :----------: |
|  `¥d `  |     任意の数字      |     [0-9]     |    -     |            -             |      -       |
|  `¥D `  |   任意の数字以外    |    [^0-9]     |    -     |            -             |      -       |
|  `¥s `  |   任意の空白文字    | [¥t¥n¥r¥f¥v]  |    -     |            -             |      -       |
|  `¥S `  | 任意の空白文字以外  | [^¥t¥n¥r¥f¥v] |    -     |            -             |      -       |
|  `¥w `  |    任意の英数字     | [a-zA-Z0-9_]  |    -     |            -             |      -       |
|  `¥W `  |  任意の英数字以外   | [¥a-zA-Z0-9_] |    -     |            -             |      -       |
|  `¥A `  |    文字列の先頭     |       ^       |    -     |            -             |      -       |
|  `¥Z `  |    文字列の末尾     |       $       |    -     |            -             |      -       |
|   `.`   |    任意の一文字     |       -       |   a.c    | abc, acc, aac abbc, accc |              |
|  ¥`^`   |    文字列の先頭     |       -       |   ^abc   |      abcdef defabc       |              |
|   `$`   |    文字列の末尾     |       -       |   abc$   |      defabc abcdef       |              |
|  `¥* `  | ０回以上の繰り返し  |       -       |  ab¥\*   | a, ab, abb, abbb aa, bb  |              |
|   `+`   | １回以上の繰り返し  |       -       |   ab+    | ab, abb, abbb a, aa, bb  |              |
|   `?`   |   ０回または１回    |       -       |   ab?    |        a, ab abb         |              |
|  `{m}`  |   m 回の繰り返し    |       -       |   a{3}   |     aaa a, aa, aaaa      |              |
| `{m,n}` | m ～ n 回の繰り返し |       -       | a{2, 4}  |  aa, aaa, aaaa a, aaaaa  |              |
|  `[] `  |        集合         |       -       |  [a-c]   |     a, b, c d, e, f      |              |
| `縦線 ` |   和集合（または    |       -       | a 縦線 b |        a, b c, d         |              |
|  `() `  |     グループ化      |       -       |  (abc)+  | abc, abcabc a, ab, abcd  |              |

### enum

列挙型

```python{cmd=true}
from enum import Enum

class Alpha(Enum):
    AA = 1
    BB = 2
    CC = 3
    DD = 4
    EE = 5

    @classmethod
    def get_member(cls):
        return dict(cls.__members__)

print(Alpha.get_member())
# {'AA': <Alpha.AA: 1>, 'BB': <Alpha.BB: 2>, 'CC': <Alpha.CC: 3>, 'DD': <Alpha.DD: 4>, 'EE': <Alpha.EE: 5>}

print(Alpha.DD.value)

print(Alpha.BB.name)

print(Alpha["EE"].value)
```

### dataclass

```python{cmd=true}
from dataclasses import dataclass, field, asdict, astuple

@dataclass
class Person():
    first_name: str = ''
    last_name: str = ''
    age = 0
    # list, set, dictはdefault_factoryを指定する
    family: list = field(default_factory=list)
    # デフォルト値を指定したいときはlambdaを使う
    country: dict = field(default_factory=lambda: {"kilometer", "mile"})

# イミュータブル
@dataclass(frozen=True)
class User:
    name: str = "boojum"
    age: int = 0

# コンストラクタで生成されるのは型アノテーションがついているクラス変数のみ
print(Person())
print(Person("Django", "Reinhardt"))

print(Person().__repr__())

print(asdict(User()))
print(astuple(User()))

# DBclassに代入。第一引数はインスタンス
PP = Person("Django", "Reinhardt")
object.__setattr__(PP, "first_name", "DJANGO")
print(PP.first_name)
```

### decimal

-   高精度の演算ができる

```python{cmd=true}
from decimal import Decimal

# floatの計算
f1 = 0.1
f2 = 0.2
f3 = f1 + f2

# decimalの計算
d1 = Decimal("0.1")
d2 = Decimal("0.2")
d3 = d1 + d2

print(f3)
print(d3)
print(f3 == d3)
```

-   decimal()の引数には文字列を指定する

```python{cmd=true}
from decimal import Decimal, getcontext, FloatOperation

# FloatOperationのtrapを有効にする
getcontext().traps[FloatOperation] = True

# float初期化で実行時エラー
d1 = Decimal(0.1)
# floatの演算で実行時エラー
d2 = Decimal(2) * 0.5
```

-   精度の指定

```python{cmd=true}
from decimal import Decimal, getcontext

# 文字列ではなく、floatを指定、FloatOperationの設定をしていないのでエラーにならない
d1 = Decimal(0.1)
d2 = Decimal(0.2)

print(d1 + d2) # 0.3000000000000000166533453694

# 精度の指定
getcontext().prec = 5
print(d1 + d2) # 0.30000
```

```python{cmd=true}
from decimal import Decimal, ROUND_HALF_UP

# default: ROUND_HALF_EVEN

d1 = Decimal("123.456")
print(d1.quantize(Decimal("0"),rounding=ROUND_HALF_UP))     # 123
# 小数部の四捨五入
print(d1.quantize(Decimal("0.1"),rounding=ROUND_HALF_UP))   # 123.5
print(d1.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))  # 123.46
print(d1.quantize(Decimal("0.001"),rounding=ROUND_HALF_UP)) # 123.456
# 整数部の四捨五入
print(d1.quantize(Decimal("1E1"),rounding=ROUND_HALF_UP))   # 1.2E+2
print(d1.quantize(Decimal("1E2"),rounding=ROUND_HALF_UP))   # 1E+2
print(d1.quantize(Decimal("1E3"),rounding=ROUND_HALF_UP))   # 0E+3
# 整数部の四捨五入(intにして見やすく)
print(int(d1.quantize(Decimal("1E1"),rounding=ROUND_HALF_UP)))   # 120
print(int(d1.quantize(Decimal("1E2"),rounding=ROUND_HALF_UP)))   # 100
print(int(d1.quantize(Decimal("1E3"),rounding=ROUND_HALF_UP)))   # 0

```

### logger

```python
from logging import getLogger, INFO, StreamHandler, Formatter

logger = getLogger(__name__)
logger.setLevel(INFO)
# StreamHandlerの設定
console_handler = StreamHandler()
console_handler.setLevel(INFO)
formatter = Formatter('%(asctime)s %(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
```

### argparse

-   パーサの宣言

```python
import argparse

parser = argparse.ArgumentParser(description="description")
```

-   パーサの追加

```python
# フラグとして使うとき
parser.add_argument("--flag", action="store_true", help="description") #"store_false"で逆
# 引数を取る時
parser.add_argument("--args", nargs="*")  # 0以上の引数を取る
parser.add_argument("--args", nargs="2")  # 固定長
parser.add_argument("--args", nargs="+")  # 1つ以上

# Namespaceオブジェクトとして渡す
args = parser.parse_args()
print(args)
```

### unittest (+pytest)

-   `fixture`

```python
@pytest.fixture
    def TestObj(self):
        Obj=Object()
        return Obj

@pytest.fixture
    def TestObj(self):
        # 前処理 SetUp
        Obj.Initialize()
        yield
        # 後処理 TearDown
        Obj.Finalize()
```

-   `parametrize`

```python
@ pytest.mark.parametrize("Param1, Param2", [
        ("FOO","BAR"),
        ("BOO","JUM"),
        ("POW","DEV"),
    ])
    def test_method(self, TestObj, Param1, Param2):
        assert TestObj.method(Param1, Param2) is None
```

-   定数の扱い

```python
#被テスト対象のクラス変数
self.const=100

#テストでは300としたい
TestObj.const=300

#Mockは使えない
TestObj.const=mock.MagicMock(return_value=300) #ERR
```

-   謎の非同期処理
    ~~非同期にするとテスト範囲が依存関係まで及ぶ。~~
    ~~async にしないとそのメソッド内でテストが完結する~~？情報が少ない。

```python
@pytest.mark.asyncio
async def test_method(self, TestObj):
    assert TestObj.method() is None
```

-   `mock`

```python
test_mock = mock.MagicMock()
with mock.patch.object(CLASS, "method", test_mock):
    test_mock.return_value = "RETURN"
    assert NONE is None

# return_value は関数の戻り値
CLASS.method = mock.MagicMock(return_value="RETURN")

# 例外発生
CLASS.method = mock.MagicMock(side_effect=Exception())
```

-   例外発生のチェック

```python
with pytest.raises(Exception):
    TestObj.method
```

-   抽象クラスのテスト

```python
@pytest.fixture
def TestObj(self):

    with mock.patch.object(
        class,
        "__abstractmethods__",
        new_callable=set
    ):
        testObj = class()

    return testObj

@pytest.mark.asyncio
async def test_method(self, TestObj):
    with pytest.raises(NotImplementedError):
        TestObj.method()
```

### pytest コンソールコマンド

-   テスト実施

> ```bash
> python -m pytest [test_FILEpath]
> ```

-   テスト情報を増やす `-v/--verbose`

> ```bash
> python -m pytest -v [test_FILEpath]
> ```

-   テスト情報を減らす `-q/--quiet`

> ```bash
> python -m pytest -q [test_FILEpath]
> ```

-   print 関数の出力を標準出力に書き出す `-s/--capture-no`

-   カバレッジ `--cov`

> ```bash
> python -m pytest --cov -v [test_FILEpath]
> ```

-   条件分岐の網羅率 `--cov --cov-branch`
-   HTML レポート出力 `--cov-report=html`

-   フォルダを限定してカバレッジを表示する

> ```bash
> python -m pytest --cov=[SOURCE CODEdir] --cov-branch -s -v [test_FILEpath]
>
> ----------- coverage: platform win32, python 3.7.7-final-0 -----------
> Name                                                       Stmts   Miss Branch BrPart  Cover
> --------------------------------------------------------------------------------------------
> [ファイル名]  [実行対象コード行数][網羅できてない行数][条件分岐の数][条件分岐の拾い溢し？][カバレッジ率]
> ```

### Radon コマンド

環境にインストール

```bash
(venv) > pip install radon
```

-   コードメトリクス計測 CC -> Cyclomatic Complexity
    循環的複雑度、可読性とか言われる。見やすくメンテしやすいコードを書こう。

```bash
radon cc -s --total-average "path¥to¥source_file"
```

[official](https://radon.readthedocs.io/en/latest/intro.html#cyclomatic-complexity)公式 HP っぽい

### py_compile

pyc ファイルを作る。難読化のためだがデコンパイルが可能らしい。

```python
>>python インタラクティブシェルで起動

>>import py_compile
>>py_compile.compile("myapp.py", "myappp.pyc") #ファイルパス、変換後の名前
```

### subprocess

pyinstaller で exe ファイルを作った時、subprocess のコンソールコマンドが動かないことがある。

> ```python
> proc = subprocess.Popen(
>    # 実行ファイルに格納する時は　sys._MEIPASS + "\\" +
>    CMD,
>    **subprocess_args(True)  # PyInstallerでNoConsole対策
> )
>
> def subprocess_args(include_stdout=True):
>    # The following is true only on Windows.
>    if hasattr(subprocess, 'STARTUPINFO'):
>        # On Windows, subprocess calls will pop up a command window by default
>        # when run from Pyinstaller with the ``--noconsole`` option. Avoid this
>        # distraction.
>        si = subprocess.STARTUPINFO()
>        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
>        # Windows doesn't search the path by default. Pass it an environment so
>        # it will.
>        env = os.environ
>    else:
>        si = None
>        env = None
>
>    # ``subprocess.check_output`` doesn't allow specifying ``stdout``::
>    #
>    #   Traceback (most recent call last):
>    #     File "test_subprocess.py", line 58, in <module>
>    #       **subprocess_args(stdout=None))
>    #     File "C:Python27libsubprocess.py", line 567, in check_output
>    #       raise ValueError('stdout argument not allowed,
>    #                           it will be overridden.')
>    #   ValueError: stdout argument not allowed, it will be overridden.
>    #
>    # So, add it only if it's needed.
>    if include_stdout:
>        ret = {'stdout': subprocess.PIPE}
>    else:
>        ret = {}
>
>    # On Windows, running this from the binary produced by Pyinstaller
>    # with the ``--noconsole`` option requires redirecting everything
>    # (stdin, stdout, stderr) to avoid an OSError exception
>    # "[Error 6] the handle is invalid."
>    ret.update({'stdin': subprocess.PIPE,
>                'stderr': subprocess.PIPE,
>                'startupinfo': si,
>                'env': env})
>    return ret
> ```

### ctype

windows コンソールで ANSI エスケープシーケンスの処理を有効にする。

```python
import ctypes

kernel32 = ctypes.windll.kernel32
kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
```

### 抽象クラス(abc)

```python
from abc import ABCMeta, abstractmethod

class InterfaceClass(metaclass=ABCMeta):
    # 継承クラスでオーバーライドしないとエラー
    @abstractmethod
    def method(self):
        raise NotImplementedError()
```

### threading

```python
import threading
import time
import datetime

start_time = datetime.datetime.now()


def func():
    while True:
        dd = datetime.datetime.now()
        print(dd)

        if dd == start_time.minute + 1:
            sub_thread.do_run = False
            break

        time.sleep(1)


sub_thread = threading.Thread(target=func)
# デーモンスレッド化
sub_thread.setDaemon(True)
# スタート
sub_thread.start()

while True:
    if not getattr(sub_thread, "do_run", True):
        break
    # 終了を待ち合わせ
    sub_thread.join(timeout=1)

# 存在するスレッドのリストを受け取る
print(threading.enumerate())
print(sub_thread in threadind.enumerate())
```

### concurrent

```python
from concurrent import futures
# 最大8プロセスで動かす
with futures.ProcessPoolExecutor(max_workers=8) as executor:
    future_list = [
        executor.submit(func, *args) for args in lists
    ]
    # 終了順にキャッチ
    for future in futures.as_completed(future_list):
        # コールバック
        pass
```

### socket

```python
def scanning(port: int):
    START = 1000
    STOP = 10000
    threads = []
    isopen = []
    def scanning(port: int):
        con = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return_code = con.connect_ex(("localhost", port))
        con.close()
        if return_code == 0:
            isopen.append(port)
    for port in range(START, STOP):
        thread = threading.Thread(target=scanning, args=(port, ))
        thread.setDaemon = True
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    for port in isopen:
        logger.info(f"PORT:{port} using")
```

### tk

-   ファイルダイアログを開く

```python
from tkinter import Tk, filedialog
root = Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
file = filedialog.askopenfilename(
    filetypes=[("データファイル", "*csv;*xlsx")]
)
# folder = filedialog.askdirectory()

return file
```

# Python launcher

Version を指定して Python を実行(Windows)

```bash
# 環境変数にあるpythonで実行
python -V

# Version指定で実行
py -3.7 -V

# 利用可能なPythonバージョンとパスのリストを表示する。*が有効
py --list-paths
```

# pip

ライブラリの書き出し、一括ダウンロード

```bash
#書き出し
>> pip freeze > requirements.text

# 読み込み
>> pip install -r requirements.text
```
