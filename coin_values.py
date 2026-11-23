"""
What this file does:
    1. Gives each recognized coin its dollar value
    2. Adds up the total value of all coins
    3. Writes the results onto the image so we can see them
"""
import cv2

# What each coin is worth (in dollars)
COIN_VALUES = {
    "nickel": 0.05,
    "dime": 0.10,
    "quarter": 0.25,
    "loonie": 1.00,
    "toonie": 2.00,
}

def get_coin_value(label):
    """Return the dollar value of ONE coin, e.g. get_coin_value("dime") -> 0.10"""
    label = label.lower()            # "Dime" and "dime" should both work

    if label in COIN_VALUES:
        return COIN_VALUES[label]
    else:
        return 0.0                   

# Add up all the coins
def calculate_total(labels):
    """
    Take a list of coin names and return two things:
        total     - the combined value in dollars
        breakdown - how many of each coin, e.g. {"dime": 2, "toonie": 1, ...}
    """
    # Start with a count of zero for every kind of coin
    breakdown = {"nickel": 0, "dime": 0, "quarter": 0, "loonie": 0, "toonie": 0}

    total_cents = 0   # we add in CENTS (whole numbers) to avoid rounding errors

    for label in labels:
        label = label.lower()
        if label in COIN_VALUES:
            breakdown[label] = breakdown[label] + 1                  
            total_cents = total_cents + round(COIN_VALUES[label] * 100)

    total = total_cents / 100     
    return total, breakdown

# Print a summary in the console
def print_report(labels):
    total, breakdown = calculate_total(labels)

    print("Coins detected:", len(labels))

    for name in breakdown:
        count = breakdown[name]
        if count > 0:                                  
            money = count * COIN_VALUES[name]
            print(f"  {name}: {count} coin(s) = ${money:.2f}")

    print(f"TOTAL VALUE: ${total:.2f}")
    return total

# Draw the results on the image
def draw_results(image, coins, total):
    """
    Draw a circle + label on every coin, and the total at the top of the image.

    image : the picture
    coins : a list of (x, y, radius, label), one entry per coin
            x, y   = the coin's centre in pixels
            radius = the coin's radius in pixels
            label  = the coin's name, e.g. "dime"
    total : the total value in dollars
    """
    result = image.copy()         # copy so we don't change the original picture

    # Colours in OpenCV are (Blue, Green, Red), NOT (Red, Green, Blue)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
  
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw each coin 
    for (x, y, radius, label) in coins:
        value = get_coin_value(label)
        text = f"{label} ${value:.2f}"         

        # Green circle around the coin
        cv2.circle(result, (x, y), int(radius), GREEN, 2)

        # Text size grows with the coin so it is readable on big and small coins
        text_size = max(0.4, radius / 60)

        # Write the label near the coin's centre
        cv2.putText(result, text, (x - int(radius), y), font, text_size, RED, 2)

    # Draw the banner at the top with the coin count and total
    image_width = result.shape[1]
    cv2.rectangle(result, (0, 0), (image_width, 40), BLACK, -1)  
    banner_text = f"Coins: {len(coins)}   Total: ${total:.2f}"
    cv2.putText(result, banner_text, (10, 28), font, 0.9, WHITE, 2)

    return result
