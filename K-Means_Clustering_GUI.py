"""Importing Libraries required"""
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

"""K-Means Class"""
"""Code for Calculating Clusters and Centroid  """
class Kmeans_Cluster:
    def __init__(self, data, columns, k, root, canvas1, tol=0.00001, max_iter=300):
        self.data = data
        self.k = int(k.get())
        self.tol = tol
        self.max_iter = max_iter
        self.columns=columns
        self.root_child = root
        self.canvas1 = canvas1
        self.arrange_data()
        self.fit() 
        
    def arrange_data(self):
        self.data['Gender'].replace(['Female','Male'],[1,2],inplace=True)
        self.data = self.data[self.columns]
        self.data= self.data.values
        
    def fit(self):
        self.centroids = {}
        for i in range(self.k):
            
            self.centroids[i] = self.data[i]
        
        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in self.data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)


            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break
        self.format_data()
        
    def format_data(self):
        self.centroids_list = [list(self.centroids[i]) for i in range(self.k)]
        self.clusters = [pd.DataFrame(self.classifications[i],columns= self.columns) for i in range(self.k)]
        self.scatter_plot()

    def scatter_plot(self):
        label3 = tk.Label(self.root_child,width=20,height=20)
        self.canvas1.create_window(25, 25, window=label3)
        
        figure1 = plt.Figure(figsize=(10,10), dpi=100)
        ax1 = figure1.add_subplot(111)
        for i in range(self.k):
            ax1.scatter(self.clusters[i][self.columns[0]], self.clusters[i][self.columns[1]],label='Cluster_'+str(i+1),s=30, alpha=0.5)
            ax1.scatter(self.centroids_list[i][0], self.centroids_list[i][1], marker='*',c='black', s=50)
        ax1.set_title('Clusters of customers')
        ax1.set_xlabel(self.columns[0])
        ax1.set_ylabel(self.columns[1])
        ax1.legend(bbox_to_anchor=(0.9, 0.7))
        scatter1 = FigureCanvasTkAgg(figure1, self.root_child)
        scatter1.get_tk_widget().pack()
        scatter1.get_tk_widget().place(height=400, width=700,relx = 0.3, rely = 0.3)


"""GUI class to create windows of application"""
class GUI:
    """Initialization"""
    def __init__(self):
        self.root = tk.Tk()
        self.widgets()
        self.root.mainloop()
    
    """Read Input Dataset"""
    def get_excel(self):
        import_file_path = filedialog.askopenfilename()
        read_file = pd.read_csv(import_file_path)
        self.data = DataFrame(read_file)
        
    """Creating Widgets to create buttons, text box"""
    def widgets(self):
        """
        """
        self.canvas1 = tk.Canvas(self.root, width = 800, height = 900,  relief = 'ridge')
        self.canvas1.pack()
        
        self.label1 = tk.Label(self.root, text='K-Means Clustering')
        self.label1.config(font=('helvetica', 24))
        self.canvas1.create_window(400, 25, window=self.label1)
        
        browseButtonExcel = tk.Button(self.root,text=" Import Input File ", command=self.get_excel, bg="red", fg="black", font=("helvetica", 14, "bold"))
        self.canvas1.create_window(400, 70, window=browseButtonExcel)
        
        self.label2 = tk.Label(self.root, text='Type Number of Clusters:')
        self.label2.config(font=('helvetica', 14))
        self.canvas1.create_window(400, 120, window=self.label2)
        
        self.entry1 = tk.Entry (self.root) 
        self.canvas1.create_window(400, 150, window=self.entry1)
        self.entry1.get()
        
        processButton = tk.Button(self.root,text=' Annual Income Cluster ', command=lambda: Kmeans_Cluster(self.data, ['Annual Income (k$)','Spending Score (1-100)'], self.entry1, self.root, self.canvas1), bg='green', fg='black', font=('helvetica', 14, 'bold'))
        self.canvas1.create_window(300, 190, window=processButton)
        processButton2 = tk.Button(self.root,text=' Age Cluster ', command=lambda: Kmeans_Cluster(self.data, ['Age','Spending Score (1-100)'], self.entry1, self.root, self.canvas1), bg='green', fg='black', font=('helvetica', 14, 'bold'))
        self.canvas1.create_window(500, 190, window=processButton2)

obj = GUI()
    
            
    
