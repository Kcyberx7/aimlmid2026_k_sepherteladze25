Finding the correlation:

Pearson Correlation Analysis and Visualization
The data points used in this analysis were extracted manually from the provided online graph. By hovering the mouse over each blue data point, the corresponding (x,y)(x, y)(x,y) coordinates were displayed on the screen and recorded. 
After collecting the data, a Python script was developed to calculate Pearson’s correlation coefficient and visualize the relationship between the two variables. The NumPy library was used to store the extracted coordinates in arrays and to compute Pearson’s correlation coefficient using the standard correlation formula. This coefficient measures the strength and direction of the linear relationship between the variables.
To support the numerical result, the Matplotlib library was used to generate a scatter plot of the data points. The scatter plot provides a visual confirmation of the linear relationship observed in the numerical analysis.
After executing the code, the resulting graph was successfully generated, showing a clear upward linear trend. The calculated Pearson correlation coefficient was approximately (real meaning r=0.9991781943735769) r=1.00, indicating a very strong positive linear correlation between the variables. 

Spam email detection:

The developed Python application consists of several logical modules. The data loading module reads the input CSV file and prepares the dataset for analysis. The model training module creates and trains a logistic regression classifier using the extracted email features. The evaluation module validates the trained model by computing the accuracy and confusion matrix on unseen test data. The visualization module generates graphical representations of the dataset and model performance. Finally, the email parsing module processes raw email text, extracts the same features used during training, and classifies the email as spam or legitimate.
To test the application, the program was executed by providing the input file k_sepherteladze25_72634.csv as a command-line argument. The dataset was successfully processed, and the application generated two graphical visualizations. Afterward, the trained model was tested using a randomly composed email text, which was classified through the interactive console interface. Screenshots of the execution results and generated visualizations are provided as uploaded files.
