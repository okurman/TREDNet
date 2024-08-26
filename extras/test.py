import sys

print("--- Before  assignment ---")
print(f"References to value_1: {sys.getrefcount('value_1')}")
print(f"References to value_2: {sys.getrefcount('value_2')}")
x = "value_1"
print("--- After   assignment ---")
print(f"References to value_1: {sys.getrefcount('value_1')}")
print(f"References to value_2: {sys.getrefcount('value_2')}")
x = "value_2"
print("--- After reassignment ---")
print(f"References to value_1: {sys.getrefcount('value_1')}")
print(f"References to value_2: {sys.getrefcount('value_2')}")
