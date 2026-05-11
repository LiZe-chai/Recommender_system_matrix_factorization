import torch
from torch import nn
from torch.optim import SGD
import math
import sys
import os
import csv
import numpy as np
import sqlite3

def create_and_load_data(database_name, train_file, test_file):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS train_data (user_id INTEGER, item_id INTEGER, rating REAL, timestamp INTEGER)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS test_data (user_id INTEGER, item_id INTEGER, timestamp INTEGER)''')
        
    with open(train_file, 'r') as train_csv:
        reader = csv.reader(train_csv)
        for row in reader:
            cursor.execute("INSERT INTO train_data VALUES (?,?,?,?)", row)
        with open(test_file, 'r') as test_csv:
            reader = csv.reader(test_csv)
            for row in reader:
                cursor.execute("INSERT INTO test_data (user_id, item_id, timestamp) VALUES (?,?,?)", (row[0], row[1], row[2]))
        conn.commit()
        conn.close()

def load_train_data(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM train_data")
    train_data = cursor.fetchall()
    conn.close()
    return train_data

def load_test_data(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test_data")
    test_data = cursor.fetchall()
    conn.close()
    return test_data

class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, num_factors):
        super(MatrixFactorization, self).__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.user_factors.weight.data.uniform_(0.25, 0.5)
        self.item_factors.weight.data.uniform_(0.25, 0.5)

    def forward(self, user_ids, item_ids):
        user_embedding = self.user_factors(user_ids)
        item_embedding = self.item_factors(item_ids)

        prediction = (user_embedding * item_embedding).sum(dim=1)  
        return prediction

def train_model_fixed_epochs(
    model,
    train_user_tensor,
    train_item_tensor,
    train_rating_tensor,
    num_epochs,
    lr=0.01,
    device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    train_user_tensor = train_user_tensor.to(device)
    train_item_tensor = train_item_tensor.to(device)
    train_rating_tensor = train_rating_tensor.to(device)

    loss_function = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr)

    num_samples = len(train_rating_tensor)
    batch_size = max(num_samples // 10, 1)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)

            batch_user = train_user_tensor[start:end]
            batch_item = train_item_tensor[start:end]
            batch_rating = train_rating_tensor[start:end]

            optimizer.zero_grad()

            predictions = model(batch_user, batch_item)
            loss = loss_function(predictions, batch_rating)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % 1000 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {avg_loss:.6f}")

    return model

def predict_rating(
    model, 
    user_id, 
    item_id, 
    user_to_idx, 
    item_to_idx,
    user_avg_dict, 
    item_avg_dict,
    global_avg,
    device=None
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    if user_id not in user_to_idx and item_id not in item_to_idx:
        return global_avg

    if user_id not in user_to_idx:
        return item_avg_dict.get(item_id, 0.5)

    if item_id not in item_to_idx:
        return user_avg_dict.get(user_id, 0.5)

    user_id_tensor = torch.tensor([user_to_idx[user_id]], dtype=torch.long).to(device)
    item_id_tensor = torch.tensor([item_to_idx[item_id]], dtype=torch.long).to(device)

    with torch.no_grad():
        prediction = model(user_id_tensor, item_id_tensor).item()

    prediction = round(prediction * 2) / 2
    prediction = max(0.5, min(prediction, 5.0))

    return prediction

def test_data_predict_ratings(
    model, 
    test_data, 
    user_to_idx, 
    item_to_idx,
    user_avg_dict, 
    item_avg_dict, 
    global_avg, 
    device):
    predictions = []
    for user_id, item_id, _ in test_data:
        rating = predict_rating(
            model, user_id, item_id,
            user_to_idx, item_to_idx,
            user_avg_dict, item_avg_dict,
            global_avg,
            device
        )
        predictions.append(rating)
    return predictions

def save_predictions_to_csv(predictions, test_data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for (user_id, item_id, timestamp), rating in zip(test_data, predictions):
            writer.writerow([user_id, item_id, rating, timestamp])

if __name__ == "__main__":
    train_file = 'train_20M.csv'
    test_file = 'test_20M.csv'
    database_name = 'task2.db'
    """ Run this if first load
    create_and_load_data(
        database_name, 
        train_file, 
        test_file
    )
    """ 
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()
    train_data = load_train_data(database_name)
    test_data = load_test_data(database_name)

    cursor.execute("SELECT DISTINCT user_id FROM train_data")
    unique_user_idx = [row[0] for row in cursor.fetchall()]

    cursor.execute("SELECT DISTINCT item_id FROM train_data")
    unique_item_idx = [row[0] for row in cursor.fetchall()]

    num_users = len(unique_user_idx)
    num_items = len(unique_item_idx)

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(unique_user_idx)
    }

    item_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(unique_item_idx)
    }
    cursor.execute(
        "SELECT user_id, AVG(rating) FROM train_data GROUP BY user_id"
    )
    train_user_tensor = torch.tensor(
        [user_to_idx[row[0]] for row in train_data],
        dtype=torch.long
    )
    train_item_tensor = torch.tensor(
        [item_to_idx[row[1]] for row in train_data],
        dtype=torch.long
    )

    train_rating_tensor = torch.tensor(
        [row[2] for row in train_data],
        dtype=torch.float32
    )

    user_avg_dict = {
        row[0]: row[1]
        for row in cursor.fetchall()
    }

    cursor.execute(
        "SELECT item_id, AVG(rating) FROM train_data GROUP BY item_id"
    )

    item_avg_dict = {
        row[0]: row[1]
        for row in cursor.fetchall()
    }

    cursor.execute("SELECT AVG(rating) FROM train_data")

    global_avg = cursor.fetchone()[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Run this if retraining the model is required.
    model = MatrixFactorization(
        num_users=num_users,
        num_items=num_items,
        num_factors=25
    )

    trained_model = train_model_fixed_epochs(
        model=model,
        train_user_tensor=train_user_tensor,
        train_item_tensor=train_item_tensor,
        train_rating_tensor=train_rating_tensor,
        num_epochs=28006,
        lr=0.01,
        device=device
    )

    torch.save({
        "model_state_dict": trained_model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "num_factors": 25
    }, "best_mf_model_full_traindata.pth")

    print("Model saved successfully.")
    """
    loaded_model_data = torch.load(
        "best_mf_model_full_traindata.pth",
        map_location=device
    )

    loaded_model = MatrixFactorization(
        loaded_model_data['num_users'],
        loaded_model_data['num_items'],
        loaded_model_data['num_factors']
    )

    loaded_model.load_state_dict(
        loaded_model_data['model_state_dict']
    )

    loaded_model = loaded_model.to(device)

    loaded_model.eval()

    predictions = test_data_predict_ratings(
        loaded_model,
        test_data,
        user_to_idx,
        item_to_idx,
        user_avg_dict,
        item_avg_dict,
        global_avg,
        device
    )

    output_file = 'Test20M_predicted_results.csv'

    save_predictions_to_csv(
        predictions,
        test_data,
        output_file
    )