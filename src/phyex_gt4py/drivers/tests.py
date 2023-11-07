

def somme(a: int, b: int, c: int, d: int):
    
    return a + b + c + d

if __name__ == "__main__":
    
    prop1 = {
        "a": 1,
        "b": 2
    }
    
    prop2 = {
        "c": 3,
        "d": 4
    }
    
    s = somme(**prop1, **prop2)
    print(s)