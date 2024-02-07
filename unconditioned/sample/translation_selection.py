import tkinter as tk
from tkinter import messagebox 
import numpy as np
class ClickCounterApp:
    def __init__(self, mc, max_x=3.0, max_y=3.0, window_width=600, window_height=600):
        self.click_count = 0
        self.points = []
        # Create the main window
        self.root = tk.Tk()
        self.max_x = max_x
        self.max_y = max_y
        self.window_width = window_width
        self.window_height = window_height
        self.root.title("Translation Selection")
        self.max_clicks = mc
        # Set the window size
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(False, False)

        # Create a canvas to capture clicks
        self.canvas = tk.Canvas(self.root, width=window_width, height=window_height, bg="white")
        self.canvas.pack()
        self.center_x = window_width // 2
        self.center_y = window_height // 2
        self.initalize()
        self.xrange = [0.1*window_width, 0.9*window_width]
        self.yrange = [0.1*window_height, 0.9*window_height]
        # Bind the click event to the on_click method
        self.canvas.bind("<Button-1>", self.on_click)
    
    def initalize(self):
        self.canvas.create_line(self.center_x, 0.9*self.window_height, self.center_x, 0.1*self.window_height, fill="black", arrow=tk.LAST)  # Vertical line
        self.canvas.create_line(0.1*self.window_width, self.center_y, 0.9*self.window_width, self.center_y, fill="black", arrow=tk.LAST)  # Horizontal line
        self.canvas.create_text(self.window_width, self.center_y +0.02*self.window_height, text=f"X Max: {self.max_x}", anchor="e")
        self.canvas.create_text(self.center_x, 0+0.08*self.window_height, text=f"Y Max: {self.max_y}", anchor="n")
        self.axis_length = 0.8*self.window_width-0.1*self.window_width
        self.root.update()
    
    def convert_point_coordinates(self, x, y):
        """
        This function converts the clicked coordinates to the actual coordinates given by max_x and max_y. A symmetrical coordinate system is assumed.
        """
        x = x - self.window_width // 2
        y = (self.window_height // 2) - y
        x = x / (self.axis_length / 2)
        y = y / (self.axis_length / 2)
        x = x * self.max_x
        y = y * self.max_y
        return x, y
    def on_click(self, event):
        x, y = event.x, event.y
        if x < self.xrange[0] or x > self.xrange[1] or y < self.yrange[0] or y > self.yrange[1]:
            messagebox.showinfo("Invalid Point", "Clicked point is outside the valid range.")
            return 
        x, y = self.convert_point_coordinates(x,y)  # Adjust coordinates to center (0,0)
        self.points.append([x, y])
        self.click_count += 1

        # Display the clicked point with its ID and coordinates
        point_id = f"Point {self.click_count}"
        point_text = f"{point_id}: x={x:.2f}, y={y:.2f}"

        # Create a text widget at the clicked position
        circle = self.canvas.create_oval(event.x - 5, event.y - 5, event.x + 5, event.y + 5, fill="blue")
        text_widget = self.canvas.create_text(event.x + 10, event.y, text=point_text, anchor="w")
        # Print the details in the console
        print(f"Clicked at position: {point_text}")

        # Update the canvas
        self.root.update()
        if self.click_count == self.max_clicks:
            # Ask for confirmation to clear the points
            confirm = messagebox.askyesno("Confirmation", "Confirm the translation points. If not they will be cleared.")
            if confirm:
                # Destroy the GUI and return the points
                self.root.destroy()
                print("RETURNING")
                return np.array(self.points)
            else:
                # Clear the canvas and reset the click count
                self.canvas.delete("all")
                self.initalize()
                self.click_count = 0


    def run(self):
        # Start the Tkinter event loop
        self.root.mainloop()
        print("GOT RETURN")
        return np.array(self.points)

"""
# Create an instance of ClickCounterApp
app = ClickCounterApp(mc=5)

# Run the application
app.run()
"""