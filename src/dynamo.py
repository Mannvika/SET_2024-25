import boto3
import uuid
import time
import random

# Initialize DynamoDB client
dynamodb = boto3.resource('dynamodb', region_name='us-east-1', aws_access_key_id='access', aws_secret_access_key='secret')  # Update region_name as needed
table = dynamodb.Table('RescueGator_Metrics') # Replace with your DynamoDB table name
# data = {"runID": "A1"}
def save_to_dynamo(data):
    """Save data to DynamoDB."""
    try:
        print("SAVING DATA")
        item = {**data}
        total_time = 0
        print("Inner Dictionaries (not YET sent to DynamoDB): ", item)
        time_dict = {}
        while total_time < 3:
            print("In the loop:")
            try:
                # Create a new time entry
                total_time += 1
                current_time_key = time.strftime("%H:%M:%S")  # Get the current time as a string key
                inner_data = [current_time_key,random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)]


                # time_dict[str(total_time)] = inner_data
                # # Add the time dictionary to the item dictionary
                # item[f"Timestamp {str(total_time)}"] = time_dict
                print(f"Timestamp {str(total_time)}: {inner_data}")
                time.sleep(1) # 1 second delay between dictionary entries

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(2)  # Wait for 2 seconds before the next iteration to prevent rapid error logs

        print("\nOuter Dictionary (NOW sent to DynamoDB):")

        item[str(total_time)] = inner_data
        print(item)

        # table.put_item(Item=item)
        return {"message": "Data saved successfully!", "item": item}
    except Exception as e:
        print(f"Error saving to DynamoDB: {e}")
        return {"message": "Error saving data", "error": str(e)}


# # used for testing
# if __name__ == "__main__":
#     save_to_dynamo(data)