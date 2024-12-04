import os
import json
import openai
import chromadb
from typing import List, Dict


# Vocareum OpenAI API Configuration
openai.api_base = "https://openai.vocareum.com/v1"
openai.api_key = "voc-20890367151266773689311674ba14d68bb69.29998898"  

class HomeMatch:
    def __init__(self):
        """
        Initialize HomeMatch with OpenAI and Chroma Vector Database
        """
        # Initialize ChromaDB client
        self.chroma_client = chromadb.Client()
        
        # Create a collection for storing property embeddings
        self.collection = self.chroma_client.create_collection(name="real_estate_listings")
        
    def generate_listings(self, num_listings: int = 10) -> List[Dict]:
        """
        Generate synthetic real estate listings using OpenAI
        """
        listings = []
        
        for _ in range(num_listings):
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise real estate listing generator. Always provide complete, detailed information."
                        },
                        {
                            "role": "user", 
                            "content": """Generate a comprehensive real estate property listing. 
                            REQUIRED FIELDS (MUST BE FILLED OUT):
                            - Neighborhood name (must be specific and unique)
                            - Exact price (with $ and comma formatting)
                            - Number of bedrooms (integer)
                            - Number of bathrooms (with decimal for half baths)
                            - Total house size in square feet (integer)
                            - Detailed property description (at least 3 sentences)
                            - Neighborhood description (at least 2 sentences)

                            Provide the response ONLY as a valid JSON with these exact keys:
                            {
                                "neighborhood": "string",
                                "price": "string",
                                "bedrooms": "integer",
                                "bathrooms": "float",
                                "houseSize": "integer",
                                "description": "string",
                                "neighborhoodDescription": "string"
                            }"""
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7,
                    response_format={"type": "json_object"}  # New in newer OpenAI models
                )
                
                # Extract and parse the listing
                listing_text = response.choices[0].message.content

                
                # Parse the JSON
                listing = json.loads(listing_text)
                
                # Validate that all required fields are present and non-empty
                required_fields = [
                    'neighborhood', 'price', 'bedrooms', 'bathrooms', 
                    'houseSize', 'description', 'neighborhoodDescription'
                ]
                
                if all(listing.get(field) for field in required_fields):
                    listings.append(listing)
                else:
                    print("Incomplete listing generated, skipping...")
            
            except Exception as e:
                print(f"Error generating listing: {e}")
        
        return listings


    def store_listings(self, listings: List[Dict], directory: str = "/home/goutham/GENAI/home"):
        """
        Store listings in .txt file at the given directory
        """
        try:
            # Create a .txt file to store the listings
            file_path = os.path.join(directory, "real_estate_listings.txt")
            
            with open(file_path, "w") as file:
                for i, listing in enumerate(listings):
                    # Convert listing to a string format
                    listing_text = f"""
                    Neighborhood: {listing.get('neighborhood', '')}
                    Price: {listing.get('price', '')}
                    Bedrooms: {listing.get('bedrooms', '')}
                    Bathrooms: {listing.get('bathrooms', '')}
                    House Size: {listing.get('houseSize', '')}
                    Description: {listing.get('description', '')}
                    Neighborhood Description: {listing.get('neighborhoodDescription', '')}
                    """
                    
                    # Write the listing to the file
                    file.write(f"Listing {i+1}:\n{listing_text}\n{'='*50}\n")
                    
            print(f"Listings have been saved to {file_path}")
            
        except Exception as e:
            print(f"Error storing listings: {e}")
   
    
    def collect_buyer_preferences(self) -> Dict:
        """
        Collect buyer preferences through interactive input
        """
        print("\n--- HomeMatch Buyer Preference Questionnaire ---")
        
        preferences = {}
        questions = [
            "Describe your ideal home size and layout",
            "What are your top 3 must-have features?",
            "Preferred neighborhood characteristics",
            "Budget range and financial considerations",
            "Lifestyle and specific location requirements"
        ]
        
        for question in questions:
            print(f"\n{question}:")
            preferences[question] = input("Your response: ")
        
        return preferences
    
    def find_matching_listings(self, preferences: Dict, top_k: int = 3):
        """
        Find matching listings using semantic search
        """
        # Convert preferences to a searchable string
        preferences_text = " ".join(preferences.values())
        
        # Generate embedding for preferences
        embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=preferences_text
        )
        
        preferences_embedding = embedding_response['data'][0]['embedding']
        
        # Perform semantic search
        results = self.collection.query(
            query_embeddings=[preferences_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def personalize_listings(self, listings: List[str], preferences: Dict):
        """
        Personalize listing descriptions
        """
        personalized_listings = []
        
        for listing in listings:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a real estate description personalizer."
                        },
                        {
                            "role": "user", 
                            "content": f"""Personalize this listing description to match these preferences:

Listing:
{listing}

Buyer Preferences:
{preferences}

Subtly emphasize aspects of the property that align with the preferences while maintaining factual accuracy."""
                        }
                    ],
                    max_tokens=300,
                    temperature=0.6
                )
                
                personalized_listing = response.choices[0].message.content
                personalized_listings.append(personalized_listing)
            
            except Exception as e:
                print(f"Error personalizing listing: {e}")
        
        return personalized_listings
    
    def run(self):
        """
        Main workflow of the HomeMatch application
        """
        # Generate initial listings
        print("Generating real estate listings...")
        listings = self.generate_listings(num_listings=10)
        
        # Store listings in .txt files in the specified directory
        print("Storing listings in .txt files...")
        self.store_listings(listings)
        
        # Add listings to the vector database with embeddings
        print("Creating embeddings for listings...")
        for listing in listings:
            # Convert listing to a searchable string
            listing_text = " ".join([
                listing.get('neighborhood', ''),
                listing.get('description', ''),
                listing.get('neighborhoodDescription', ''),
                str(listing.get('price', '')),
                str(listing.get('bedrooms', '')),
                str(listing.get('bathrooms', ''))
            ])
            
            # Generate embedding for the listing
            embedding_response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=listing_text
            )
            
            listing_embedding = embedding_response['data'][0]['embedding']
            
            # Add the listing to the collection
            self.collection.add(
                embeddings=[listing_embedding],
                documents=[listing_text],
                ids=[f"listing_{listings.index(listing)}"]
            )
        
        # Collect buyer preferences
        print("\nLet's find your perfect home!")
        preferences = self.collect_buyer_preferences()
        
        # Find matching listings
        print("\nSearching for matching listings...")
        matched_listings = self.find_matching_listings(preferences)
        
        # Personalize matched listings
        print("\nPersonalizing matched listings...")
        personalized_listings = self.personalize_listings(matched_listings, preferences)
        
        # Display results
        print("\n--- Personalized Matching Listings ---")
        for i, listing in enumerate(personalized_listings, 1):
            print(f"\nListing {i}:\n{listing}\n{'='*50}")
def main():
    # Create and run HomeMatch application
    home_match = HomeMatch()
    home_match.run()

if __name__ == "__main__":
    main()
