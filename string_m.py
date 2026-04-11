import time


def extract_domain(email: str):
    a = email.find("@")
    print(email[a + 1 :])


def ex_ext(file_names: list[str]):
    # ext = [file[len(file) - file[::-1].find(".") :] for file in file_names]
    ext = [
        file.split(".")[-1] for file in file_names
    ]  ## the same as above faster with fewer . ( best overall )
    return ext


fi = ["image..sd.s.ds.d.s.d..s.d.s.d.s.d.s.d.s.d.pdf", "s.s..s.png"]


start_time = time.time()

i = 0
ext: list[str] = []
# while i < 1000000:
ext += ex_ext(fi)
print(ext)


# print(time.time() - start_time)


# print(ext)
