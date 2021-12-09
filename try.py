class A:
    def a(self):
        raise NotImplementedError

class B(A):
    def a(self):
        return 1

class C(A):
    def a(self):
        return 2

class D(C, B):
    ...

if __name__ == '__main__':
    print(D().a())