"""Seed the database with an admin user and a demo food catalogue.

Run directly:  ``python -m database.seed_data``  or  ``python database/seed_data.py``

The script is idempotent: running it again refreshes food documents and
re-indexes ChromaDB without creating duplicates.
"""

import re
import sys
from pathlib import Path

# Allow running as a plain script (``python database/seed_data.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auth import auth_service  # noqa: E402
from config import settings  # noqa: E402
from database import mongo  # noqa: E402
from services import food_service  # noqa: E402
from utils.date_utils import utcnow  # noqa: E402


FOODS = [
    ("Chicken Momo", "Snack", "Nepali", "non_veg", ["lunch", "snack", "dinner"], "medium", 250,
     ["chicken", "flour", "onion", "spices"], ["gluten"], ["dumpling", "popular", "affordable", "steamed"],
     "Chicken Momo is a popular Nepali steamed dumpling filled with minced chicken and spices."),
    ("Veg Momo", "Snack", "Nepali", "veg", ["lunch", "snack"], "mild", 180,
     ["cabbage", "carrot", "flour", "onion"], ["gluten"], ["dumpling", "vegetarian", "steamed", "affordable"],
     "Veg Momo is a steamed dumpling filled with fresh mixed vegetables."),
    ("Buff Momo", "Snack", "Nepali", "non_veg", ["lunch", "snack", "dinner"], "spicy", 220,
     ["buff", "flour", "onion", "spices"], ["gluten"], ["dumpling", "spicy", "popular"],
     "Buff Momo is a spicy Nepali steamed dumpling filled with minced buffalo meat."),
    ("Chicken Chowmein", "Main", "Chinese", "non_veg", ["lunch", "dinner"], "medium", 300,
     ["chicken", "noodles", "vegetables", "soy sauce"], ["gluten", "soy"], ["noodles", "stir-fry", "popular"],
     "Chicken Chowmein is stir-fried noodles with chicken and vegetables."),
    ("Veg Chowmein", "Main", "Chinese", "veg", ["lunch", "dinner"], "mild", 220,
     ["noodles", "cabbage", "carrot", "soy sauce"], ["gluten", "soy"], ["noodles", "vegetarian", "affordable"],
     "Veg Chowmein is stir-fried noodles tossed with fresh vegetables."),
    ("Chicken Biryani", "Main", "Indian", "non_veg", ["lunch", "dinner"], "spicy", 380,
     ["chicken", "rice", "spices", "yogurt"], ["dairy"], ["rice", "aromatic", "spicy", "filling"],
     "Chicken Biryani is fragrant basmati rice cooked with spiced chicken."),
    ("Veg Biryani", "Main", "Indian", "veg", ["lunch", "dinner"], "medium", 300,
     ["rice", "mixed vegetables", "spices"], [], ["rice", "vegetarian", "aromatic"],
     "Veg Biryani is aromatic rice cooked with mixed vegetables and spices."),
    ("Paneer Butter Masala", "Main", "Indian", "veg", ["lunch", "dinner"], "medium", 350,
     ["paneer", "butter", "tomato", "cream", "spices"], ["dairy"], ["creamy", "rich", "vegetarian", "popular"],
     "Paneer Butter Masala is soft cottage cheese in a rich creamy tomato gravy."),
    ("Dal Bhat Tarkari", "Main", "Nepali", "veg", ["lunch", "dinner"], "mild", 280,
     ["rice", "lentils", "vegetables"], [], ["traditional", "healthy", "wholesome", "staple"],
     "Dal Bhat Tarkari is the traditional Nepali meal of rice, lentil soup, and vegetable curry."),
    ("Chicken Thakali Set", "Main", "Nepali", "non_veg", ["lunch", "dinner"], "medium", 450,
     ["rice", "chicken", "lentils", "vegetables", "pickle"], [], ["traditional", "set-meal", "filling"],
     "Chicken Thakali Set is a hearty Nepali platter with rice, chicken curry, lentils, and sides."),
    ("Veg Thakali Set", "Main", "Nepali", "veg", ["lunch", "dinner"], "mild", 380,
     ["rice", "lentils", "vegetables", "pickle"], [], ["traditional", "set-meal", "vegetarian"],
     "Veg Thakali Set is a traditional Nepali platter with rice, lentils, and vegetable curries."),
    ("Mushroom Pizza", "Main", "Italian", "veg", ["lunch", "dinner"], "mild", 550,
     ["mushroom", "cheese", "flour", "tomato"], ["gluten", "dairy"], ["cheesy", "baked", "vegetarian"],
     "Mushroom Pizza is a cheesy baked pizza topped with fresh mushrooms."),
    ("Margherita Pizza", "Main", "Italian", "veg", ["lunch", "dinner"], "mild", 500,
     ["cheese", "flour", "tomato", "basil"], ["gluten", "dairy"], ["cheesy", "classic", "vegetarian"],
     "Margherita Pizza is a classic pizza with tomato, mozzarella, and basil."),
    ("Chicken Burger", "Fast Food", "Fast Food", "non_veg", ["lunch", "snack", "dinner"], "medium", 320,
     ["chicken", "bun", "lettuce", "mayo"], ["gluten", "egg"], ["burger", "filling", "popular"],
     "Chicken Burger is a juicy grilled chicken patty in a soft bun with fresh veggies."),
    ("Veg Burger", "Fast Food", "Fast Food", "veg", ["lunch", "snack"], "mild", 250,
     ["potato", "bun", "lettuce", "mayo"], ["gluten"], ["burger", "vegetarian", "affordable"],
     "Veg Burger is a crispy vegetable patty in a soft bun with fresh salad."),
    ("Spicy Chicken Burger", "Fast Food", "Fast Food", "non_veg", ["lunch", "snack", "dinner"], "spicy", 450,
     ["chicken", "bun", "chili", "mayo"], ["gluten", "egg"], ["burger", "spicy", "filling"],
     "Spicy Chicken Burger is a fiery crispy chicken burger with a spicy sauce."),
    ("Chicken Fried Rice", "Main", "Chinese", "non_veg", ["lunch", "dinner"], "medium", 300,
     ["rice", "chicken", "egg", "vegetables"], ["egg", "soy"], ["rice", "stir-fry", "popular"],
     "Chicken Fried Rice is wok-fried rice with chicken, egg, and vegetables."),
    ("Egg Fried Rice", "Main", "Chinese", "egg", ["lunch", "dinner"], "mild", 240,
     ["rice", "egg", "vegetables", "soy sauce"], ["egg", "soy"], ["rice", "affordable", "quick"],
     "Egg Fried Rice is fried rice tossed with scrambled egg and vegetables."),
    ("Veg Fried Rice", "Main", "Chinese", "veg", ["lunch", "dinner"], "mild", 220,
     ["rice", "vegetables", "soy sauce"], ["soy"], ["rice", "vegetarian", "affordable"],
     "Veg Fried Rice is fried rice with a colourful mix of vegetables."),
    ("Spicy Ramen", "Main", "Chinese", "non_veg", ["lunch", "dinner"], "very_spicy", 420,
     ["noodles", "chicken", "chili", "egg"], ["gluten", "egg", "soy"], ["noodles", "soupy", "very-spicy"],
     "Spicy Ramen is a hot noodle soup with chicken, egg, and a fiery chili broth."),
    ("Chicken Sekuwa", "Snack", "Nepali", "non_veg", ["snack", "dinner"], "spicy", 400,
     ["chicken", "spices", "garlic"], [], ["grilled", "smoky", "spicy", "popular"],
     "Chicken Sekuwa is smoky Nepali-style grilled chicken marinated in spices."),
    ("Aloo Paratha", "Breakfast", "Indian", "veg", ["breakfast", "lunch"], "mild", 150,
     ["potato", "flour", "spices"], ["gluten"], ["breakfast", "stuffed", "vegetarian", "affordable"],
     "Aloo Paratha is a flatbread stuffed with spiced mashed potatoes."),
    ("Masala Dosa", "Breakfast", "Indian", "veg", ["breakfast", "snack"], "medium", 200,
     ["rice", "lentils", "potato", "spices"], [], ["breakfast", "crispy", "south-indian", "vegetarian"],
     "Masala Dosa is a crispy rice crepe filled with spiced potato masala."),
    ("Samosa Chaat", "Snack", "Indian", "veg", ["snack"], "spicy", 160,
     ["potato", "flour", "chickpeas", "chutney"], ["gluten"], ["street-food", "tangy", "spicy", "vegetarian"],
     "Samosa Chaat is crushed samosa topped with chickpeas, yogurt, and tangy chutneys."),
    ("Caesar Salad", "Salad", "Italian", "veg", ["lunch", "snack"], "mild", 300,
     ["lettuce", "cheese", "croutons", "dressing"], ["gluten", "dairy"], ["salad", "light", "healthy"],
     "Caesar Salad is a light salad of crisp lettuce, cheese, and croutons."),
    ("Veg Soup", "Soup", "Chinese", "veg", ["dinner", "snack"], "mild", 180,
     ["vegetables", "corn starch"], [], ["soup", "light", "warm", "healthy"],
     "Veg Soup is a light, warm broth packed with fresh vegetables."),
    ("Fresh Fruit Salad", "Salad", "Continental", "vegan", ["breakfast", "lunch", "snack"], "mild", 220,
     ["seasonal fruits", "mint", "lemon juice"], [], ["salad", "cold", "fresh", "light", "healthy", "cooling"],
     "Fresh Fruit Salad is a chilled bowl of seasonal fruits with mint and a squeeze of lemon."),
    ("Mango Lassi", "Drink", "Indian", "veg", ["lunch", "snack"], "mild", 150,
     ["mango", "yogurt", "sugar"], ["dairy"], ["drink", "cold", "sweet", "refreshing", "cooling"],
     "Mango Lassi is a chilled, creamy yogurt drink blended with sweet mango."),
    ("Fresh Lemon Soda", "Drink", "Continental", "vegan", ["lunch", "snack", "dinner"], "mild", 120,
     ["lemon", "soda water", "mint"], [], ["drink", "cold", "fizzy", "refreshing", "cooling"],
     "Fresh Lemon Soda is an ice-cold fizzy lemon drink with fresh mint."),
    ("Vanilla Ice Cream", "Dessert", "Continental", "veg", ["snack", "dinner"], "mild", 180,
     ["milk", "cream", "sugar", "vanilla"], ["dairy"], ["dessert", "cold", "sweet", "ice cream", "cooling"],
     "Vanilla Ice Cream is two scoops of creamy, chilled vanilla ice cream."),
]

# Demo today's specials: dish name -> discount percent.
SPECIALS = {
    "Chicken Momo": 15,
    "Spicy Ramen": 20,
}


# Nutrition per serving: dish name -> (calories kcal, protein g). Powers the
# "something light" / "high protein" / "filling" recommendations.
NUTRITION = {
    "Chicken Momo": (450, 18),
    "Veg Momo": (320, 8),
    "Buff Momo": (480, 22),
    "Chicken Chowmein": (550, 24),
    "Veg Chowmein": (380, 8),
    "Chicken Biryani": (720, 32),
    "Veg Biryani": (580, 12),
    "Paneer Butter Masala": (650, 22),
    "Dal Bhat Tarkari": (620, 18),
    "Chicken Thakali Set": (780, 35),
    "Veg Thakali Set": (640, 16),
    "Mushroom Pizza": (700, 26),
    "Margherita Pizza": (680, 25),
    "Chicken Burger": (600, 28),
    "Veg Burger": (480, 14),
    "Spicy Chicken Burger": (640, 30),
    "Chicken Fried Rice": (620, 26),
    "Egg Fried Rice": (540, 16),
    "Veg Fried Rice": (480, 10),
    "Spicy Ramen": (560, 24),
    "Chicken Sekuwa": (520, 30),
    "Aloo Paratha": (400, 9),
    "Masala Dosa": (380, 8),
    "Samosa Chaat": (450, 9),
    "Caesar Salad": (280, 12),
    "Veg Soup": (120, 4),
    "Fresh Fruit Salad": (150, 2),
    "Mango Lassi": (260, 7),
    "Fresh Lemon Soda": (90, 0),
    "Vanilla Ice Cream": (320, 5),
}


def _image_for(name: str) -> str:
    """Photo filename for a dish, when the file is bundled in assets/foods."""
    filename = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") + ".jpg"
    return filename if (settings.FOOD_IMAGE_DIR / filename).is_file() else None


def seed_admin() -> None:
    """Create the admin account from the configured credentials (if missing)."""
    existing = mongo.users().find_one({"email": settings.ADMIN_EMAIL})
    if existing:
        print(f"Admin already exists: {settings.ADMIN_EMAIL}")
        return
    now = utcnow()
    mongo.users().insert_one(
        {
            "name": "Administrator",
            "email": settings.ADMIN_EMAIL,
            "password_hash": auth_service.hash_password(settings.ADMIN_PASSWORD),
            "role": "admin",
            "is_active": True,
            "preferences": {
                "preferred_cuisines": [],
                "diet_type": None,
                "spice_level": None,
                "usual_budget": None,
                "disliked_ingredients": [],
                "allergies": [],
            },
            "created_at": now,
            "updated_at": now,
            "last_login": None,
        }
    )
    print(f"Created admin user: {settings.ADMIN_EMAIL} / {settings.ADMIN_PASSWORD}")


def seed_foods() -> None:
    """Insert or refresh the demo food catalogue and index it in ChromaDB."""
    print("Seeding foods...")
    for row in FOODS:
        (name, category, cuisine, diet, meal, spice, price,
         ingredients, allergens, tags, description) = row
        data = {
            "name": name,
            "category": category,
            "cuisine": cuisine,
            "diet_type": diet,
            "meal_type": meal,
            "spice_level": spice,
            "price": price,
            "ingredients": ingredients,
            "allergens": allergens,
            "tags": tags,
            "description": description,
            "available": True,
            "is_special": name in SPECIALS,
            "discount_percent": SPECIALS.get(name, 0),
            "image": _image_for(name),
            "calories": NUTRITION.get(name, (None, None))[0],
            "protein": NUTRITION.get(name, (None, None))[1],
        }
        existing = mongo.foods().find_one({"name": name})
        if existing:
            food_service.update_food(str(existing["_id"]), data)
        else:
            food_service.add_food(data)
    print(f"Seeded {len(FOODS)} foods and indexed them in ChromaDB.")


def main() -> None:
    """Seed the admin account and the demo menu, then index them in ChromaDB."""
    print("Connecting to MongoDB...")
    if not mongo.ping():
        print(
            "ERROR: Could not connect to MongoDB. "
            "Start MongoDB (mongodb://localhost:27017) or set MONGODB_URI in .env."
        )
        sys.exit(1)

    seed_admin()
    seed_foods()
    print("\nSeed complete. You can now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
