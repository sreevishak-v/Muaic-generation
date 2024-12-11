import streamlit as st
import tensorflow as tf
import pretty_midi
import numpy as np
import pandas as pd
import base64

# Load the pre-trained model
model = tf.keras.models.load_model('music_generation_model.keras')

def generate_music(seed_notes, num_predictions, temperature):
    # Convert seed notes from text to a numerical array
    if seed_notes:
        try:
            # Convert comma-separated string into a list of integers
            seed_list = seed_notes.split(',')
            # Convert to array of [pitch, velocity, duration] for each note
            seed_array = np.array([[int(seed_list[i]), 100, 1.0] for i in range(len(seed_list))])
        except ValueError:
            st.error("Invalid seed notes format! Use a comma-separated list of integers.")
            return None
    else:
        # Use a default seed if no input is provided
        seed_array = np.random.randint(50, 70, size=(10, 3))  # Default seed notes of length 10

    # Pad seed_array to ensure it has 20 timesteps, each with 3 features
    while len(seed_array) < 20:
        # Pad with zeros to make it 20 timesteps (you can also repeat the last note instead of zero padding)
        seed_array = np.concatenate([seed_array, np.zeros((1, 3))], axis=0)

    # Reshape seed_array to match the expected shape (1, 20, 3)
    seed_array = np.reshape(seed_array, (1, 20, 3))  # (1, 20, 3)

    # Generate predictions
    generated_notes = seed_array.tolist()  # Start with the seed
    for _ in range(num_predictions):
        input_sequence = generated_notes[-20:]  # Get the last 20 notes
        input_sequence = np.reshape(input_sequence, (1, 20, 3))  # Ensure correct shape (1, 20, 3)
        
        predictions = model.predict(input_sequence, verbose=0)
        predictions = predictions / temperature  # Adjust predictions based on temperature
        probabilities = tf.nn.softmax(predictions[0]).numpy()  # Softmax to get probability distribution
        next_note = np.random.choice(len(probabilities), p=probabilities)  # Sample the next note
        generated_notes.append(next_note)

    return generated_notes


# Function to convert generated notes into MIDI and display the audio
def display_audio_from_notes(generated_notes):
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()

    # Create an Instrument object for the generated notes (e.g., piano)
    piano = pretty_midi.Instrument(program=pretty_midi.program_to_instrument_number('Acoustic Grand Piano'))

    # Convert generated notes to MIDI
    for note in generated_notes:
        pitch, velocity, duration = note
        midi_note = pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=0, end=duration)
        piano.notes.append(midi_note)

    # Add the instrument to the PrettyMIDI object
    midi.instruments.append(piano)

    # Save MIDI to a file
    midi_file = 'generated_music.mid'
    midi.write(midi_file)

    # Display audio in Streamlit
    with open(midi_file, "rb") as f:
        audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode()
        audio_url = f"data:audio/midi;base64,{audio_b64}"
        st.markdown(f'<audio controls><source src="{audio_url}" type="audio/midi"></audio>', unsafe_allow_html=True)

# Streamlit app
st.title("Music Generation App")

# Input parameters
seed_notes = st.text_area("Enter seed notes (optional)", "")
num_predictions = st.slider("Number of predictions", 100, 1000, 200)
temperature = st.slider("Temperature", 0.1, 2.0, 1.0)

# Generate and display music
if st.button("Generate Music"):
    generated_notes = generate_music(seed_notes, num_predictions, temperature)
    if generated_notes is not None:
        display_audio_from_notes(generated_notes)

# Download MIDI file
if 'generated_notes' in locals() and generated_notes is not None:
    midi_data = generated_notes  # Assuming your notes are in the correct format
    midi_file = 'generated_music.mid'
    # Save the generated MIDI to file
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=pretty_midi.program_to_instrument_number('Acoustic Grand Piano'))

    for note in generated_notes:
        pitch, velocity, duration = note
        midi_note = pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=0, end=duration)
        piano.notes.append(midi_note)

    midi.instruments.append(piano)
    midi.write(midi_file)

    # Convert the MIDI file to a downloadable link
    with open(midi_file, "rb") as f:
        audio_data = f.read()
        audio_b64 = base64.b64encode(audio_data).decode()
        audio_url = f"data:audio/midi;base64,{audio_b64}"
        href = f'<a href="{audio_url}" download="generated_music.mid">Download MIDI</a>'
        st.markdown(href, unsafe_allow_html=True)
