def CtoF(c: float) -> float:
    f: float = c * (9 / 5) - 32
    return f


def FtoC(f: float) -> float:
    c: float = 5 / 9 * (f - 32)
    return c


def main():
    usr_in = input("Enter the degree with the unit (c/f): ")
    v = 0.0
    m = "N"
    if len(usr_in) < 2:
        print("Not enough input")
        return
    if usr_in[-1].lower() == "c":
        v = CtoF(int(usr_in[:-1]))
        m = "f"
    elif usr_in[-1].lower() == "f":
        v = FtoC(int(usr_in[:-1]))
        m = "c"
    else:
        print("You didn't provided a valid unit")
        return

    print(f"{v}{m}")


if __name__ == "__main__":
    main()
