import matplotlib.pyplot as plot
import numpy as np

x = [1, 100]
y = [1, 10]

plot.plot(x, y)
plot.title("Linear")
plot.xlabel("X-axis")
plot.ylabel("Y-axis")
plot.show()


x2 = np.linspace(0, 2 * np.pi, 1000)

y1 = 2 * np.cos(x2)
y2 = np.cos(x2)
y3 = 0.5 * np.cos(x2)
y4 = np.sin(x2)

plot.plot(x2, y1, "r--", label="2cos(x)")
plot.plot(x2, y2, "b-", label="cos(x)")
plot.plot(x2, y3, "g:", label="0.5cos(x)")
plot.plot(x2, y4, "r-", label="sin(x)")

plot.xlabel("x")
plot.ylabel("y")
plot.title("Trigonometric Functions")
plot.legend()

plot.show()


# // (stop - start)/( num-1 )
