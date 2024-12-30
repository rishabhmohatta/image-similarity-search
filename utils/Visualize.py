import numpy as np
import matplotlib.pyplot as plt

def plot_results(query_image,distance,x_train,y_train,indices):
    num_queries = query_image.shape[0]
    for i in range(num_queries):
        plt.figure(figsize=(10, 5))
        
        # Display the original image
        plt.subplot(1, len(indices[i]) + 1, 1)
        plt.imshow(query_image[i].squeeze(),cmap='gray')
        plt.title("Query Image")
        plt.axis('off')
        
        # Plot the similar images
        for j in range(len(indices[i])):
            plt.subplot(1, len(indices[i]) + 1, j + 2)
            plt.imshow(x_train[indices[i][j]].squeeze(),cmap='gray')
            plt.title(f"cls:{y_train[indices[i][j]]},(Dist:{distance[i][j]:.2f})")
            plt.text(0.5, -0.2, f"Img:{j+1}", ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    
        plt.tight_layout()
        plt.savefig(f"images\image_{i}_out.png")
        plt.show()
        
