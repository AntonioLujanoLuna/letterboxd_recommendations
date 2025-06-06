#!/usr/local/bin/python3.11

from pymongo.operations import ReplaceOne, UpdateOne
import requests
from bs4 import BeautifulSoup
import pymongo
from pymongo.errors import BulkWriteError
from pprint import pprint
from tqdm import tqdm
import time
import random
from db_connect import connect_to_db

# Connect to MongoDB client
db_name, client, tmdb_key = connect_to_db()
db = client[db_name]
users = db.users

def scrape_user_section(url_template, section_name, max_pages=128):
    """Scrape users from a specific section of Letterboxd"""
    users_data = []
    
    pbar = tqdm(range(1, max_pages + 1), desc=f"Scraping {section_name}")
    for page in pbar:
        try:
            r = requests.get(url_template.format(page))
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", attrs={"class": "person-table"})
            
            if not table:
                break
                
            rows = table.findAll("td", attrs={"class": "table-person"})
            
            for row in rows:
                link = row.find("a")["href"]
                username = link.strip('/')
                display_name = row.find("a", attrs={"class": "name"}).text.strip()
                
                # Extract review count
                small_text = row.find("small").find("a").text.replace('\xa0', ' ')
                num_reviews = int(small_text.split()[0].replace(',', ''))
                
                users_data.append({
                    "username": username,
                    "display_name": display_name,
                    "num_reviews": num_reviews,
                    "user_category": section_name
                })
            
            # Add small delay to be respectful
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue
    
    return users_data

def get_diverse_users():
    """Scrape users from different sections to get diversity"""
    all_users = []
    
    # Different user categories for diversity
    sections = [
        # Most active users (original approach)
        {
            "url": "https://letterboxd.com/members/popular/this/week/page/{}/",
            "name": "popular_this_week",
            "pages": 50
        },
        # All-time popular (mix of old and new users)
        {
            "url": "https://letterboxd.com/members/popular/page/{}/",
            "name": "popular_all_time",
            "pages": 50
        },
        # Recently joined active users (fresh perspectives)
        {
            "url": "https://letterboxd.com/members/popular/this/month/page/{}/",
            "name": "popular_this_month",
            "pages": 30
        },
        # Users with crew picks (quality-focused users)
        {
            "url": "https://letterboxd.com/crew/popular/this/all-time/page/{}/",
            "name": "crew_picks",
            "pages": 20
        }
    ]
    
    for section in sections:
        users_data = scrape_user_section(
            section["url"], 
            section["name"], 
            section["pages"]
        )
        all_users.extend(users_data)
        print(f"Scraped {len(users_data)} users from {section['name']}")
    
    return all_users

def filter_users_by_activity(users_list):
    """Create a balanced set of users across activity levels"""
    # Sort by number of reviews
    sorted_users = sorted(users_list, key=lambda x: x['num_reviews'])
    
    # Define activity buckets
    buckets = {
        'casual': [],      # 10-50 reviews
        'regular': [],     # 50-200 reviews
        'active': [],      # 200-500 reviews
        'power': [],       # 500-1000 reviews
        'super': []        # 1000+ reviews
    }
    
    for user in sorted_users:
        reviews = user['num_reviews']
        if 10 <= reviews < 50:
            buckets['casual'].append(user)
        elif 50 <= reviews < 200:
            buckets['regular'].append(user)
        elif 200 <= reviews < 500:
            buckets['active'].append(user)
        elif 500 <= reviews < 1000:
            buckets['power'].append(user)
        elif reviews >= 1000:
            buckets['super'].append(user)
    
    # Sample from each bucket
    balanced_users = []
    samples_per_bucket = {
        'casual': 1000,
        'regular': 2000,
        'active': 3000,
        'power': 3000,
        'super': 1000
    }
    
    for bucket_name, bucket_users in buckets.items():
        sample_size = min(len(bucket_users), samples_per_bucket[bucket_name])
        sampled = random.sample(bucket_users, sample_size)
        balanced_users.extend(sampled)
        print(f"{bucket_name}: {len(sampled)} users (from {len(bucket_users)} available)")
    
    return balanced_users

def main():
    print("Starting diverse user collection...")
    
    # Get users from different sections
    all_users = get_diverse_users()
    print(f"Total users scraped: {len(all_users)}")
    
    # Remove duplicates based on username
    unique_users = {user['username']: user for user in all_users}.values()
    print(f"Unique users: {len(unique_users)}")
    
    # Balance across activity levels
    balanced_users = filter_users_by_activity(list(unique_users))
    print(f"Balanced user set: {len(balanced_users)}")
    
    # Prepare bulk update operations
    update_operations = []
    for user in balanced_users:
        user['user_diversity_score'] = calculate_diversity_score(user)
        update_operations.append(
            UpdateOne(
                {"username": user["username"]},
                {"$set": user},
                upsert=True
            )
        )
    
    # Bulk write to database
    try:
        if len(update_operations) > 0:
            # Write in batches of 1000
            for i in range(0, len(update_operations), 1000):
                batch = update_operations[i:i+1000]
                users.bulk_write(batch, ordered=False)
                print(f"Wrote batch {i//1000 + 1}")
    except BulkWriteError as bwe:
        pprint(bwe.details)
    
    print(f"Successfully updated {len(balanced_users)} diverse users")

def calculate_diversity_score(user):
    """Calculate a diversity score for sampling"""
    # Prioritize users with moderate activity (not too few, not too many reviews)
    reviews = user['num_reviews']
    if 50 <= reviews <= 500:
        activity_score = 1.0
    elif reviews < 50:
        activity_score = 0.5
    else:
        activity_score = 0.7
    
    # Bonus for specific categories
    category_scores = {
        'crew_picks': 1.2,
        'popular_this_month': 1.1,
        'popular_this_week': 1.0,
        'popular_all_time': 0.9
    }
    category_score = category_scores.get(user.get('user_category', ''), 1.0)
    
    return activity_score * category_score

if __name__ == "__main__":
    main()