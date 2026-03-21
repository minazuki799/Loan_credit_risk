import requests
from pprint import pprint

base_url = "https://pokeapi.co/api/v2/"


def get_pokemon_info(pokemon_name):
    url = f"{base_url}/pokemon/{pokemon_name}"
    response = requests.get(url)

    if response.status_code == 200:
        print("Data retrieved successfully!")
        data = response.json()
        if data:
            print(f"Name: {data['name']}")
            print(f"ID: {data['id']}")
            print(f"Height: {data['height']}")
            print(f"Weight: {data['weight']}")

    else:
        print("failed to retriecve data for the specified Pokémon.")


pokemon_name = "pikachu"
get_pokemon_info(pokemon_name)
