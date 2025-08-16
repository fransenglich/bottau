
l = [2, 1, 3]

print(sorted(l))

print(l)

l.sort()

l.remove(1)
print(l)

#l.append(l.copy())
print(l)

print(l.index(3))

l.extend(l)
print(l)
l.clear()

my_list = [5, 10, 15, 20, 25]
print(my_list[::2])
print(my_list[::3])
print(my_list[2::])
print(my_list[2::2])

print(my_list[::-1])
reversed(my_list)

print(my_list[1:1])


my_list = [7, 14, 21, 28, 35]

print(my_list[:2:2])

my_list = [10, 20, 30, 40, 50]

print(my_list[1::2])

print(my_list[0:])

my_list = [10, 20, 30, 40, 50, 60]
print(my_list[-3:])
print(my_list[-3:6])
print(my_list[3:])

items = [1, 2, 3, 4]
for item in items[::-1]:
    print(item)

for i in range(len(items) -1, -1, -1):
    print(items[i])

if not "x" in my_list:
    print("Doesn't Exist")

numbers = [2, 4, 6, 8, 10]
if not 7 in numbers:
    print("Missing")

l = [x ** 2 for x in range(1, 6)]

print(l)

numbers = [1, 2, 3, 4]
doubled = [x * 2 for x in numbers]
print(doubled)

multiples_of_three = [x for x in range(1, 11) if x % 3 == 0]
print(multiples_of_three)

words = ["hello", "world"]
uppercase_words = [w.upper() if w == "" else w.upper() for w in words ]
print(uppercase_words)

print([0] * 3)

print([[[0 for j in range(2)] for l in range(2)] for k in range(2)])

matrix = [[i for j in range(4)] for i in range(4)]

print(matrix)

print([[1, 2, 3]] * 2)
print([3] * 2)
#print([[1, 2, 3]])
#print([[[[1, 2, 3]]]])

t = (1, 2, 3, 4)
print(t[:2])

d = {'a': 1, 'b': 2, 'c': 3}

print(d.get('d'))
print(d.pop('e', 3.14))

print(d)
d.popitem()
print(d)
print(d.values())

for value in d:
    print(value)

d.keys()

#for (k, v) in d:
    #print(k, v)

if 'a' in d:
    print("Foo")

print(d)
for i in d:
    print(i)

if 'key' not in d:
    print("NO EXCEPTION")

s = "asd"#
#s[1] = 'f'

if "abc" in "defabc":
    print("MATCH")

s.replace("d", "D")
print(s)


s = """A
B
C"""

print(s)

s = "       AB      CDE C     asdas   "
print(s.count("CC"))
print(s.split())
#s.isdigit(s)

print(s.strip())

foo = ""
if foo == None:
    pass

x = 10

def show():
    global  x
    x = x + 5
    print(x)

show()

def f():
    return 1, 2

one, _ = f()

print(one)
print(_)