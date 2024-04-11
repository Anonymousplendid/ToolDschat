import random
from copy import deepcopy

fewshots = [
    {
        "question": "Human: I planted some sunflowers and they are already drooping a bit. What did I do wrong?",
        "answer": """Response 1: Sunflowers are sun-loving flowers, and a lot of people think that they\u2019re best grown in warm weather climates.  It can be done in northern areas, and sometimes they take a while to get established, but they do like full sun to bloom.  So if your plants are sun-loving plants, they probably want to grow in more sun and less shade, and more warm weather.
Response 2: You may have planted them in the wrong kind of soil.  Sunflowers grow best in sandy soils, rather than the clay you probably have.

Evaluation: In this conversation, the response should aim to troubleshoot the drooping sunflowers. Response 1 provides actionable advice on sunlight and warmth requirements, making it helpful and practical. Response 2 prematurely assumes soil issues, lessening its helpfulness. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": "Human: What kind of Christmas decorations can I make out of felt?\n\nAssistant: Felt isn\u2019t a really common material to use for Christmas decorations, but you could probably just buy some felt from the craft store.  You could use it to make your own Christmas tree decoration or paper chains or some other ornament.  You might also be able to use it to make felt ornaments, but if you want to do that you would want to coat them in some water-based material to make them less fragile.  My guess is that you\u2019d also want to give your decorations as gifts, which is probably not a priority for most people these days.\n\nHuman: I think it is pretty common.\n\nAssistant: Oh, I guess it is. Well if you\u2019re going to make felt Christmas decorations for gifts, I recommend just buying them instead of making them. You\u2019ll get better looking decorations.\n\nHuman: I want to have fun making them.\n\nAssistant: You could make some in the shape of stockings or socks.\n\nHuman: Sounds good. Any other ideas?",
        "answer": """Response 1: I don\u2019t think so, though I\u2019m no expert on felt crafts.
Response 2: I don\u2019t know. Maybe a snowflake, a reindeer?

Evaluation: In this conversation, the response should aim to provide creative suggestions for felt Christmas decorations. Response 1 is dismissive and lacks helpfulness, while Response 2 offers creative ideas, aligning with helpfulness and practicality. Therefore, the preferred response is Response 2.""",
    },
    {
        "question": "Human: Are older cars less safe to drive?\n\nAssistant: Older cars may not be as safe as newer ones, although they are often maintained better.  Safety features are added to vehicles from model year to model year, and the quality of the vehicles themselves tends to improve over the years as well.  But many older cars were made before the safety standards were as stringent as they are today, and they may not be as crash-proof.\n\nHuman: Are air bags a lot safer to have than not?",
        "answer": """Response 1: Air bags are a safety feature added to many vehicles starting in the early 1990s.  They have helped make cars safer, and they also make the cars themselves safer, since they\u2019ve helped to reduce the number of crashes the vehicles are involved in.
Response 2: Air bags are designed to help protect the passengers in a car in the event of a crash.  If the air bags work correctly, they can help reduce the risk of injury to the driver and passenger.  They are standard on all new vehicles, and are also available as an option on many older vehicles.  So having air bags can help you be safer on the road, and also in the event of a crash.

Evaluation: In this conversation, the response should aim to inform about vehicle safety. Response 1 inaccurately implies air bags reduce the number of crashes, confusing their purpose, which impacts the honesty and sophistication of the information. Response 2 correctly explains the function of air bags and their impact on safety, providing a more comprehensive and practical answer. Therefore, the preferred response is Response 2.""",
    },
    {
        "question": "Human: How do I make a wreath?\n\nAssistant: To make a wreath, start by gathering materials. You will need green and/or artificial foliage, artificial berries, wooden skewers, wire, and glue. Lay out the foliage in a circle, then place the wooden skewers in the center of the circle. String the wire through the holes in the skewers, then weave the foliage around the wires to form a wreath shape. Once your wreath is finished, add a bow and glue on your artificial berries to create a beautiful holiday decoration.\n\nHuman: What kind of wire do I use?",
        "answer": """Response 1: To make a wreath, you will need wire that is at least 3\u20448 inch in diameter, which is also known as 18-gauge wire. You can buy wire at most hardware stores or craft stores. In a pinch, you can also use string or wire hangers.
Response 2: I would use wire that is made of a material known as \u201cmetal\u201d. Metal wire has the qualities of strength and conductivity, and can be made from materials such as steel, copper, silver, gold, platinum, aluminum, zinc, and iron.

Evaluation: In this conversation, the response should aim to provide specific guidance on choosing wire for wreath-making. Response 1 gives specific and actionable advice for selecting wire, aligning with helpfulness and practicality. Response 2 is too general and not tailored to wreath-making, lacking practicality. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": 'Human: I\'m going to Italy and need to learn some of the language.\n\nAssistant: I\'m happy to help!  How would you like to approach learning?\n\nHuman: I would like to learn some weather words. What is "rain" in Italian?\n\nAssistant: It\'s pioggia. And "snow" is neve.\n\nHuman: Perfect! What about "thunder" and "lightning"?',
        "answer": """Response 1: That's tuono and fulmine.
Response 2: Thunder is tuono. Lightning is fulmine. And thunder in Italian sounds like \"ta-rump-ta-rump\", and lightning like \"car

Evaluation: In this conversation, the response should aim to offer Italian translations for weather terms. Response 1 is clear and accurate, meeting the goal of helpfulness. Response 2 adds unnecessary detail that may confuse. Therefore, the preferred response is Response 1.""",
    },
    {
        "question": "Human: What are some festivals I can go to like Woodstock?\n\nAssistant: Woodstock is a festival that took place on a farm in New York in 1969.\n\nHuman: Right. So it's not around anymore. I just want similar festivals.",
        "answer": """Response 1: There are similar music festivals in a number of places, like Boston, Colorado, France, Spain, London, Canada, Australia, and many more.
Response 2: One festival that comes to mind is Burning Man, which is a music and art festival that happens every year in Black Rock City, Nevada.

Evaluation: In this conversation, the response should aim to suggest festivals akin to Woodstock. Response 1 is vague, while Response 2 provides a specific festival, enhancing its usefulness. Therefore, the preferred response is Response 2.""",
    },
]

prompt = """Here's a dialogue between a human and an AI assistant. The AI's mission is to provide accurate, timely, and relevant information, and to understand and respond to inquiries with detailed, reliable, and neutral information. Evaluate the AI's responses with respect to the following principles and specific criteria:

Helpfulness: The response should directly address the user's question or stated need.
Politeness: The response should maintain a respectful and considerate tone throughout.
Honesty: The response must be truthful and transparent, avoiding misleading information.
Sophistication: The response should demonstrate a depth of knowledge and articulate complex ideas clearly when necessary.
Emotional Awareness: The response should be sensitive to the emotional tone of the human and respond appropriately.
Humility: The response should acknowledge the limitations of the AI's knowledge and capabilities when applicable.
Practicality: The response should provide actionable information or direct the user to where they can find the assistance they need.

Please evaluate each of the AI assistant's responses in this dialogue, considering how well it achieves these objectives. Provide a brief justification for your evaluation of each response."""


def get_prompt():
    fewshots_now = deepcopy(fewshots)
    random.shuffle(fewshots_now)
    ret = prompt + "\n\n"
    for fewshot in fewshots_now:
        ret += (
            "---\n-- dialogue begin --\n"
            + fewshot["question"]
            + "\n"
            + "-- dialogue end --\n\n"
        )
        ret += fewshot["answer"]
