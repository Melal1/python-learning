def extract_domain(email):
    # TODO: Find the @ symbol and extract the domain part
    email.find("@")
    return ""


print(extract_domain("john.doe@example.com"))


def extract_formats(file_names):
    # TODO: Extract file extensions from the list of file names
    pass


# Test the function
file_names = ["document1.txt", "image.jpg", "presentation.pptx"]
extract_formats(file_names)

