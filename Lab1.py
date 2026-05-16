import matplotlib.pyplot as plt
import numpy as np

# Generate x values

x = np.linspace(0, 2 * np.pi, 10)
print(x)

# TODO: Define y1, y2, and y3 using cosine functions

y1 = 2 * np.cos(x)


# TODO: Plot the graphs with different styles

plt.plot(x, y1, "r--", label="2cos(x)")

# Add labels and title to the plot
plt.xlabel("x")
plt.ylabel("y")
plt.title("Graphs of Trigonometric Functions")
plt.legend()

# Show the plot
plt.show()


def fahrenheit_to_centigrade():
    # TODO: Take user input and convert Fahrenheit to Centigrade
    fahrenheit = float(input("Enter temperature in Fahrenheit: "))
    pass


fahrenheit_to_centigrade()
