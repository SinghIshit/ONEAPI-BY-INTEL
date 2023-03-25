import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Define a function to get the data from the Pokemon API
def get_pokemon_data():
    url = "https://pokeapi.co/api/v2/pokemon"
    data = requests.get(url).json()
    return data['results']

# Define a function to retrieve the base stats of a given Pokemon
def get_pokemon_stats(pokemon_url):
    pokemon_stats = requests.get(pokemon_url).json()['stats']
    stats_dict = {}
    for stat in pokemon_stats:
        stat_name = stat['stat']['name']
        base_stat = stat['base_stat']
        stats_dict[stat_name] = base_stat
    return stats_dict

# Define the app layout
st.title("Pokemon Battle Simulator")
st.write("Enter the names of two players and select a Pokemon for each player to simulate a battle:")

# Get the data from the Pokemon API
pokemon_data = get_pokemon_data()

# Add two text input fields for the user to enter the names of the players
player1_name = st.text_input("Player 1 name:")
player2_name = st.text_input("Player 2 name:")

# Add two dropdown menus for the user to select a Pokemon for each player
player1_pokemon = st.selectbox("{}'s Pokemon".format(player1_name), [pokemon['name'].capitalize() for pokemon in pokemon_data],key="1")
player2_pokemon = st.selectbox("{}'s Pokemon".format(player2_name), [pokemon['name'].capitalize() for pokemon in pokemon_data],key="2")

# Get the base stats for each player's Pokemon
player1_url = [pokemon['url'] for pokemon in pokemon_data if pokemon['name'] == player1_pokemon.lower()][0]
player1_stats = get_pokemon_stats(player1_url)
player2_url = [pokemon['url'] for pokemon in pokemon_data if pokemon['name'] == player2_pokemon.lower()][0]
player2_stats = get_pokemon_stats(player2_url)

# Convert the stats of each player's Pokemon to a pandas DataFrame
player1_stats_df = pd.DataFrame.from_dict(player1_stats, orient='index', columns=['Player 1'])
player2_stats_df = pd.DataFrame.from_dict(player2_stats, orient='index', columns=['Player 2'])
stats_df = pd.concat([player1_stats_df, player2_stats_df], axis=1)

# Determine the winner of the battle based on the total base stats of each player's Pokemon
player1_total_stats = sum(player1_stats.values())
player2_total_stats = sum(player2_stats.values())
if player1_total_stats > player2_total_stats:
    st.write("{} wins the battle!".format(player1_name))
elif player2_total_stats > player1_total_stats:
    st.write("{} wins the battle!".format(player2_name))
else:
    st.write("It's a tie!")

# Add a bar chart to visualize the base stats for each player's Pokemon
st.write("Base stats comparison for {} vs {}".format(player1_pokemon, player2_pokemon))
fig, ax = plt.subplots()
if player1_total_stats>player2_total_stats:
    ax.bar(stats_df.index, stats_df['Player 1'], color='r', label=player1_name)
    ax.bar(stats_df.index, stats_df['Player 2'], color='b', label=player2_name)
else:
    ax.bar(stats_df.index, stats_df['Player 2'], color='b', label=player2_name)
    ax.bar(stats_df.index, stats_df['Player 1'], color='r', label=player1_name)
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)


