# Heritige Housing

# Contents
1. [Design](#design)
2. [User Stories]()
3. [Dataset Content](#dataset-content)
4. [Business Requirements](#business-requirements)
5. [Hypothesis and validations](#hypothesis-and-how-to-validate)
6. [ML business case](#machine-learning-business-case)
7. [Dashboard Design](#dashboard-design)
8. [Deployment](#deployment)
9. [Unfixed Bugs](#unfixed-bugs)
10. [Fixed Bugs](#fixed-bugs)
11. [Testing](#testing)
12. [contributing](#contributing)
13. [Media](#media)
14. [Credits](#credits)
15. [Acknowledgements](#acknowledgements-optional)

## How to use this repo

1. Use this template to create your GitHub project repo

2. Log into the cloud-based IDE with your GitHub account.

3. On your Dashboard, click on the Create button

4. Paste in the URL you copied from GitHub earlier

5. Click Create

6. Wait for the workspace to open. This can take a few minutes.

7. Open a new terminal and `pip3 install -r requirements.txt`

11. Open the jupyter_notebooks directory and click on the notebook you want to open.

12. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.8.18 as it inherits from the workspace so it will be Python-3.8.18 as installed by our template. To confirm this you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with *Regenerate API Key*.

## Design

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

# Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate?

There is more than 1 hypothesis to consider with this. See below for my hypothesis and the validations
### Hypothesis 1: Larger Houses Have Higher Sale Prices

**Reasoning:** Generally, larger homes tend to sell for more because they offer more living space, which is a key factor in determining property value.
**Validation:** This can be validated by analyzing the correlation between the total square footage (including above-ground living area and basement) and the sale price. A positive correlation would support this hypothesis. We will create scatter plots and compute correlation coefficients to visualize and quantify the relationship.

### Hypothesis 2: Location Significantly Affects House Prices

**Reasoning:** The location of a house often has a significant impact on its price, with certain neighborhoods or proximity to amenities being more desirable and, therefore, more expensive.
**Validation:** This hypothesis can be validated by examining the correlation between the sale price and categorical variables representing location, such as the neighborhood or proximity to schools and parks. We can use box plots to compare sale prices across different neighborhoods and calculate ANOVA or t-tests to determine if the differences are statistically significant.
### Hypothesis 3: Houses with Higher Quality and Condition Ratings Have Higher Sale Prices

**Reasoning:** Homes that are well-maintained and built with high-quality materials are likely to command higher prices.
**Validation:** This can be validated by analyzing the correlation between sale price and variables related to quality and condition, such as overall quality (OverallQual), exterior quality (ExterQual), and overall condition (OverallCond). We will use scatter plots and correlation coefficients to measure the strength of these relationships.

### Hypothesis 4: Newer Houses Are Priced Higher Than Older Houses

**Reasoning:** Newer homes often have modern amenities, better insulation, and require less immediate maintenance, making them more attractive to buyers.
**Validation:** This hypothesis can be validated by examining the correlation between the year built (YearBuilt) and the sale price. We will analyze the relationship using scatter plots and correlation coefficients to see if newer homes consistently sell for more.

### Hypothesis 5: Houses with Additional Features (e.g., Pools, Garages, Fireplaces) Have Higher Sale Prices

**Reasoning:** Additional features like swimming pools, garages, and fireplaces add value to a home and are often reflected in a higher sale price.
**Validation:** This hypothesis can be validated by analyzing the correlation between sale price and the presence of additional features (e.g., PoolArea, GarageArea, FireplaceQu). We will use box plots and correlation coefficients to assess the impact of these features on the sale price.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

Business Requirement 1: Understanding How House Attributes Correlate with Sale Price
Objective: The client wants to identify which attributes of houses are most strongly correlated with their sale prices to understand what factors drive property values in Ames, Iowa.

### Potential Data Visualizations:

* Correlation Heatmap: A heatmap displaying the correlation coefficients between various house attributes (e.g., size, location, quality) and the sale price. This provides a quick overview of which factors are most strongly correlated with house prices.

* Scatter Plots: Individual scatter plots for key continuous variables (e.g., total square footage, year built) against sale price. This will help in visualizing the linear relationships and any potential outliers.

* Box Plots: Box plots for categorical variables (e.g., neighborhood, house style) against sale price. This will allow the client to see how sale prices vary across different categories and identify which categories are associated with higher or lower prices.
Histograms/Bar Charts: Histograms for continuous variables (e.g., living area, lot size) and bar charts for categorical variables (e.g., exterior quality, overall condition) to show the distribution of these attributes in the dataset.

Machine Learning Tasks:

Feature Importance Analysis: Using a machine learning model like Random Forest or Gradient Boosting to assess the importance of different features in predicting sale price. This will quantitatively show which attributes have the most impact on the price.
Regression Analysis: Multiple linear regression to quantify the relationships between house attributes and sale price, providing insights into how much each feature contributes to the price.
Rationale:

The visualizations help the client intuitively understand the relationships between house attributes and sale price, while the machine learning tasks provide a more rigorous and quantitative analysis of these relationships.

Business Requirement 2: Predicting House Sale Prices
Objective: The client wants to accurately predict the sale prices of the four inherited houses and any other house in Ames, Iowa, to ensure they are priced competitively and in line with the market.

### Potential Data Visualizations:

* Prediction Error Plot: A plot comparing predicted sale prices versus actual sale prices for the test dataset. This will help visualize how well the model is performing and where it might be over- or under-predicting.

* Residual Plots: Residual plots to analyze the difference between predicted and actual prices. This helps in understanding the distribution of errors and identifying any systematic biases in the predictions.

### Machine Learning Tasks:

Model Training: Training various regression models (e.g., Linear Regression, Ridge Regression, Random Forest, Gradient Boosting) to predict sale prices based on the identified significant features.

Model Validation: Evaluating the performance of the trained models using metrics like R-squared, RMSE, and MAE on a test dataset. This ensures the model generalizes well to unseen data.
Hyperparameter Tuning: Fine-tuning the model parameters to optimize performance, ensuring the most accurate predictions possible.

Final Model Deployment: Once the best model is selected, it will be used to predict the sale prices of the four inherited houses.

### Rationale:

* Accurate predictions are crucial for the client to set the right price for the inherited properties. The visualizations offer a clear view of the model's performance, while the machine learning tasks focus on building and validating a model that can reliably predict sale prices.
Summary

* Data Visualizations provide intuitive, easy-to-understand insights into the relationships between house attributes and sale prices, helping to identify key factors that affect pricing.
Machine Learning Tasks focus on building, validating, and fine-tuning predictive models that can accurately estimate house prices based on the identified features, ensuring that the client can make informed pricing decisions for the inherited properties.

## Machine Learning Business Case
### Business Objective
The client needs to maximize the sale prices of four inherited properties in Ames, Iowa. Given her familiarity with property prices in her own state, she is concerned about accurately pricing these properties in a different market. The client seeks a data-driven approach to understand the factors that influence house prices in Ames and to predict the sale prices of these properties accurately.

### Problem Statement
The client requires an accurate prediction of the sale prices for the four inherited houses in Ames, Iowa, to ensure they are competitively priced in the market. Incorrect pricing, whether too high or too low, could result in significant financial losses—either through missed sales opportunities or undervaluing the properties.

### ML Solution Overview
We propose developing a predictive model using machine learning techniques to estimate house sale prices in Ames, Iowa. The model will be trained on historical house sale data from Ames, incorporating various house attributes (e.g., size, location, quality, year built) to learn patterns and relationships that determine sale prices. This model will then be used to predict the sale prices of the inherited properties.

### Expected Benefits
Accurate Pricing: By predicting sale prices using a data-driven model, the client can ensure that the inherited properties are priced accurately, maximizing potential revenue.
Market Understanding: The model will provide insights into which features most influence house prices in Ames, helping the client better understand this new market.
Competitive Advantage: Accurate pricing will give the client a competitive edge in the Ames real estate market, potentially leading to quicker sales and higher returns.
Key Metrics for Success
R-squared (R²): Measures the proportion of variance in the sale price that can be explained by the model. A higher R² indicates a better fit of the model.
Root Mean Squared Error (RMSE): Provides a measure of the average error in predicting the sale price. Lower RMSE values indicate more accurate predictions.
Mean Absolute Error (MAE): Reflects the average magnitude of errors in the predictions. Lower MAE values indicate better model performance.
Risks and Considerations
Data Quality: Incomplete or inaccurate data could lead to poor model performance. Careful data cleaning and preprocessing are essential.
Overfitting: The model might perform well on training data but poorly on unseen data. This risk will be mitigated through cross-validation and regularization techniques.
Market Changes: The model is based on historical data and may not fully account for future market changes or trends, so ongoing model updates might be necessary.
Implementation Plan
Data Collection and Preprocessing: Gather and clean the historical house sale data from Ames, Iowa, ensuring all relevant features are included.
Exploratory Data Analysis (EDA): Perform EDA to understand the data distribution, correlations, and potential outliers.
Model Selection and Training: Train several regression models (e.g., Linear Regression, Random Forest, Gradient Boosting) using the data, selecting the model with the best performance.
Model Validation and Tuning: Validate the model using a separate test dataset, and fine-tune the model's hyperparameters to optimize performance.
Prediction and Reporting: Use the final model to predict the sale prices of the inherited houses and generate a report summarizing the findings and recommendations for pricing.
Monitoring and Maintenance: Regularly monitor the model's performance and update it as needed to reflect any market changes.

### Cost-Benefit Analysis

Costs: Time and resources for data collection, model development, and maintenance.
Benefits: Increased revenue from accurately priced properties, reduced risk of financial losses due to incorrect pricing, and enhanced understanding of the Ames real estate market.
### Conclusion
Implementing a machine learning model to predict house prices will empower the client to price her inherited properties competitively, maximizing potential returns while minimizing the risk of pricing errors. The model will serve as a robust tool to navigate the unfamiliar Ames real estate market confidently.

## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.
## Potential Problems
As i may have to connect to the Code institute custom database, there may be an issue with seeing the application incase the CI database has any issues or potential problems which could cause me to not be able to get all of the application built.
## Fixed Bugs

## Testing
This section is all about the tests i have performed to the application manually and via different browsers.
### Manual Testing

### Browser Testing

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

## Contributing

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)
- My mentor precious Ijege
