<br>

# よく使うグラフ

---

<br>
<br>
<br>

---

<!-- code_chunk_output -->

-   [よく使うグラフ](#よく使うグラフ)
    -   [折れ線グラフ](#折れ線グラフ)
    -   [棒グラフ](#棒グラフ)
    -   [円グラフ](#円グラフ)
    -   [ヒストグラム](#ヒストグラム)
    -   [レーダーチャート](#レーダーチャート)
    -   [散布図](#散布図)
    -   [パレート図](#パレート図)
    -   [クラス図 (mermaid 使用)](#クラス図-mermaid-使用)
    -   [シーケンス図 (mermaid 使用)](#シーケンス図-mermaid-使用)
    -   [フローチャート (mermaid 使用)](#フローチャート-mermaid-使用)
    -   [樹枝図 (mermaid 使用)](#樹枝図-mermaid-使用)
    -   [状態遷移図](#状態遷移図)
    -   [箱ひげ図](#箱ひげ図)
    -   [等値線図](#等値線図)
    -   [アローダイアグラム](#アローダイアグラム)
    -   [サイクル図](#サイクル図)
    -   [流線図](#流線図)
    -   [クラス図](#クラス図)
    -   [ER 図](#er-図)

<!-- /code_chunk_output -->

> matplotlib、mermaid、gnuplot など。(gnuplot はインストールしてない。)

<br>
<br>
<br>
<br>

---

---

## 折れ線グラフ

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(5)
y = x**2

plt.plot(x, y)
plt.show()
```

## 棒グラフ

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(3)
y = np.array([100, 30, 70])
plt.bar(x, y)
plt.show()
```

---

## 円グラフ

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.array([12, 23, 100])
str_label = ['a', 'b', 'c']
plt.pie(x, labels=str_label, counterclock=False, startangle=90)
plt.show()
```

```mermaid
pie
  "iOS": 45.2
  "iPhone": 17.2
  "PHP": 8.6
  "Objective-C": 6.5
  "Swift": 6.5
  "Xcode": 4
  "Laravel": 3
  "Realm": 3
  "Android": 3
  "Others": 2
```

---

## ヒストグラム

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(50, 10, 100)
plt.hist(x)
plt.show()
```

---

## レーダーチャート

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

labels = ['a', 'b', 'c', 'd']
values = [10, 20, 30, 40]
angles = np.linspace(0, 2*np.pi, len(labels) + 1, endpoint=True)
values = np.concatenate((values, [values[0]]))

fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(angles, values, 'o-')
ax.fill(angles, values, alpha=0.25)
ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
plt.show()
```

---

## 散布図

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(10)
y = np.random.rand(10)
plt.scatter(x, y)
plt.show()
```

---

## パレート図

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(4)
y1 = np.array([10, 5, 3, 2])
sum_y1 = np.sum(y1)
y2 = np.cumsum(y1) / sum_y1

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.bar(x, y1)
ax2.plot(x, y2, c='r')
plt.show()
```

---

## クラス図 (mermaid 使用)

```mermaid
classDiagram

Class01 <|-- AveryLongClass : Cool
Class03 *-- Class04
Class05 o-- Class06
Class07 .. Class08
Class09 --> C2 : Where am i?
Class09 --* C3
Class09 --|> Class07
Class07 : equals()
Class07 : Object[] elementData
Class01 : size()
Class01 : int chimp
Class01 : int gorilla
Class08 <--> C2: Cool label
```

---

## シーケンス図 (mermaid 使用)

```mermaid
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail...
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
```

---

## フローチャート (mermaid 使用)

```mermaid
graph TB
  Macの選び方 --> 持ち歩く
  持ち歩く -->|はい| スペック
  持ち歩く -->|いいえ| 予算
  スペック -->|必要| R1[MacBook Pro]
  スペック -->|低くても良い| R2[MacBook Air]
  予算 --> |いくらでもある| R3[Mac Pro]
  予算 --> |できれば抑えたい| R4[Mac mini / iMac]
```

---

##　ガントチャート (mermaid 使用)

```mermaid
gantt
dateFormat  YYYY-MM-DD
title Adding GANTT diagram to mermaid

section A section
Completed task            :done,    des1, 2014-01-06,2014-01-08
Active task               :active,  des2, 2014-01-09, 3d
Future task               :         des3, after des2, 5d
Future task2               :         des4, after des3, 5d
```

---

## 樹枝図 (mermaid 使用)

```mermaid
graph LR
    A>開始] --> |"順調"|B[終わって暇];
    A -.-> C{Dはよ};
    A --> D(実装が遅れた);
    B --> F;
    C --> F;
    D ==> E((Dはよ));
    D ==> C;
    E --> F{>>デスマ<<};
    click B "https://www.bing.com/" "リンクもはれる";
```

---

## 状態遷移図

```mermaid
stateDiagram
  [*] --> 待機
  待機 --> [*]
  待機 --> 索敵
  索敵 --> 待機
  索敵 --> 威嚇射撃
  威嚇射撃 --> [*]
```

```mermaid
stateDiagram
  plan --> Do
  Do --> Check
  Check --> Act
  Act --> plan

state Check{
  checkit --> check : OK
  check --> do : OK
  check --> NG : NG
}
```

---

## 箱ひげ図

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

spread = np.random.rand(50) * 100
center = np.ones(25) * 50
flier_high = np.random.rand(10) * 100 + 100
flier_low = np.random.rand(10) * -100
data = np.concatenate((spread, center, flier_high, flier_low))

plt.boxplot(data)
plt.show()
```

---

## 等値線図

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

delta = 0.025
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2

plt.contour(X, Y, Z)
plt.show()
```

---

## アローダイアグラム

```mermaid
graph TD

1 -->|1 day| 2
2 -->|2 days| 3
2 -->|4 days| 4
3 -->|3 days| 5
4 -->|3 hours| 5
```

---

## サイクル図

```mermaid
graph LR

P-->D
D-->C
C-->A
A-->P
```

---

## 流線図

```python {cmd matplotlib}
import numpy as np
import matplotlib.pyplot as plt

X = [1, 3]
Y = [2, 4]
U = [5, 60]
V = [70, 8]
plt.quiver(X, Y, U, V)
plt.show()
```

---

## クラス図

```mermaid
classDiagram
  Component <|-- ConcreteComponent
  Component <|-- Decorator
  Decorator o-- Component
  Component : +operation()
  ConcreteComponent : +operation()
  Decorator : -component
  Decorator : +operation()
  Decorator <|-- ConcreteDecorator
  ConcreteDecorator : +operation()
```

```mermaid
classDiagram
  classA --|> classB : Inheritance
  classC --* classD : Composition
  classE --o classF : Aggregation
  classG --> classH : Association
  classI -- classJ : Link(Solid)
  classK ..> classL : Dependency
  classM ..|> classN : Realization
  classO .. classP : Link(Dashed)

  Customer "1" --> "*" Ticket
  Student "1" --> "1..*" Course
  Galaxy --> "many" Star : Contains
```

```mermaid
classDiagram
class Shape{
    <<interface>>
    noOfVertices
    draw()
}
class Color{
    <<enumeration>>
    RED
    BLUE
    GREEN
    WHITE
    BLACK
}
```

| Type  |  Description  |
| :---: | :-----------: |
|  <--  |  Inheritance  |
| \*--  |  Composition  |
|  o--  |  Aggregation  |
|  -->  |  Association  |
|  --   | Link (Solid)  |
|  ..>  |  Dependency   |
| ..\|> |  Realization  |
|  ..   | Link (Dashed) |

## ER 図

```mermaid
erDiagram
    CAR ||--o{ NAMED-DRIVER : allows
    CAR {
        string registrationNumber
        string make
        string model
    }
    PERSON ||--o{ NAMED-DRIVER : is
    PERSON {
        string firstName
        string lastName
        int age
    }

```

| Value (left) | Value (right) |            Meaning            |
| :----------: | :-----------: | :---------------------------: |
|     \|o      |      o\|      |          Zero or one          |
|     \|\|     |     \|\|      |          Exactly one          |
|      }o      |      o{       | Zero or more (no upper limit) |
|     }\|      |      \|{      | One or more (no upper limit)  |

```mermaid
%%{init: { 'logLevel': 'debug', 'theme': 'forest' } }%%
gitGraph
    commit id: "Alpha"
    commit id: "Beta"
    commit id: "Gamma"
    commit
    commit
    branch develop
    checkout develop
    commit id: "Reverse" type: REVERSE
    commit
    checkout main
    merge develop
    commit id: "Highlight" type: HIGHLIGHT
    commit id: "Normal" tag: "v1.0.0"
```

```plantuml
#plantumlはJAVEを環境変数に
@startuml
Alice -> Bob: Authentication Request
Bob --> Alice: Authentication Response

Alice -> Bob: Another authentication Request
Alice <-- Bob: another authentication Response
@enduml
```
