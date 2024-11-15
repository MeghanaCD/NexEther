import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
data = pd.read_csv('amazon datasets.csv')  # Adjust the path if needed

# Define the recommendation function
def get_recommendations(product_name):
    # Check if the product exists in the dataset
    if product_name not in data['product_name'].values:
        return None

    # Using TF-IDF Vectorizer to convert product names into numerical form
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['product_name'])

    # Compute cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Get the index of the product that matches the name
    idx = data[data['product_name'] == product_name].index[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar products
    sim_scores = sim_scores[1:6]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top 5 most similar products with their prices and reviews
    return data.iloc[product_indices][['product_name', 'discounted_price', 'review_content']]

# Main function to run the recommendation system
if __name__ == "__main__":
    product_name = input("Enter the product name: ")
    recommendations = get_recommendations(product_name)

    if recommendations is not None:
        print("\nRecommendations:")
        for index, row in recommendations.iterrows():
            print(f"\nProduct: {row['product_name']}")
            print(f"Price: {row['discounted_price']}")
            print(f"Review: {row['review_content']}\n")
    else:
        print("Product not found. Please check the product name.")
