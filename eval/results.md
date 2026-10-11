# Probabilistic Monkey benchmark results

- Multiple choice: **17/20 (85%)**, required 75%
- Open-ended: **14/15 (93%)**, required 60%
- Automatic verdict: **PASS**
- Runtime: 36s on CPU with the NumPy engine

## Multiple choice

| result | question | model's pick | correct |
|---|---|---|---|
| PASS | Which part of a computer executes the instructions of a program? | The CPU | The CPU |
| FAIL | Which kind of memory loses its contents when the computer is switched off? | A hard disk | RAM |
| PASS | How many bits make up one byte? | Eight | Eight |
| PASS | Which of these is an operating system? | Linux | Linux |
| PASS | Which of these is an example of an input device? | A keyboard | A keyboard |
| PASS | What number system do computers use internally? | Binary | Binary |
| PASS | What is the job of DNS on the internet? | It converts domain names into IP addresses | It converts domain names into IP addresses |
| FAIL | Which device forwards packets from one network to another network? | A webcam | A router |
| PASS | Which protocol makes sure data arrives reliably and in the correct order? | TCP | TCP |
| PASS | Which of these is a valid IPv4 address? | 192.168.0.1 | 192.168.0.1 |
| PASS | What does the S in HTTPS mean? | Secure | Secure |
| PASS | What is the meaning of the abbreviation LAN? | Local area network | Local area network |
| PASS | What are the small pieces that data is broken into when it travels across a network? | Packets | Packets |
| PASS | Which default port is used by HTTPS? | 443 | 443 |
| PASS | Which of these best describes a firewall? | It filters network traffic using rules | It filters network traffic using rules |
| PASS | Which protocol automatically hands out IP addresses to devices that join a network? | DHCP | DHCP |
| PASS | Which technology connects devices to a network using radio waves instead of cables? | Wi-Fi | Wi-Fi |
| PASS | Which protocol is better for a live video call where speed matters more than perfect reliability? | UDP | UDP |
| FAIL | In a client server setup, what is your web browser? | The server | The client |
| PASS | Which description of the internet is correct? | A global network of networks | A global network of networks |

## Open-ended (greedy decoding)

**PASS - In a few words, what does a computer's CPU do?**

> The CPU, or central processing unit, is the brain of a computer. It carries out the instructions of programs by fetching, decoding and executing them, billions of times per second.

**PASS - Explain what RAM is used for.**

> RAM, or random access memory, is the computer's fast short-term memory. It holds the programs and data the CPU is using right now. RAM is volatile, so its contents are lost when the power is turned off.

**PASS - Could you describe what an IP address is?**

> An IP address is a unique number that identifies a device on a network, like a postal address for a computer. It is used to send data to the right device.

**PASS - Tell me what a router is for.**

> A router forwards packets between different networks. It reads the destination IP address of each packet and sends it along the best path. A home router connects your local network to the internet.

**PASS - Briefly, what does the Domain Name System do?**

> The Domain Name System (DNS) is a global network service that translates domain names like example.com into IP addresses, allowing multiple computers to communicate with each other. It works by using DNS servers and other internet-connected devices to find the IP address of a specific computer on the internet.
> 
> When you type a website address into your browser, the DNS translates it into an IP address that the computer can understand. The DNS acts as a phone book for the internet, translating domain names into IP addresses that the computer can use to communicate with other computers in the network

**PASS - How do TCP and UDP differ?**

> TCP is reliable: it makes a connection, resends lost packets and keeps them in order, so it is used for web pages, email and downloads. UDP is faster but does not guarantee delivery, so it is used for live video, games and voice calls.

**PASS - Explain what a computer network is.**

> A computer network is a group of computers and devices connected together so they can share resources, communicate with each other, and access data and services. It's like a group chat where everyone shares their thoughts and ideas.

**PASS - What job does a firewall do on a network?**

> A firewall is a security system that monitors network traffic and blocks or allows it according to rules. It acts as a barrier between a trusted network and the internet.

**PASS - How would you tell hardware apart from software?**

> Software is a collection of instructions that tell the hardware what to do, like a web browser or a game. Software is usually written in programming languages such as Python, JavaScript, or C++.

**PASS - Why is HTTPS safer than HTTP?**

> HTTPS is the secure version of HTTP. It encrypts the data between your browser and the website using TLS, so other people cannot read or change it. Plain HTTP is not encrypted.

**FAIL - Describe a byte in one sentence.**

> A byte is a small piece of data sent over a network.

**PASS - What is an operating system? Give an example.**

> An operating system is the main software that manages a computer. It controls the hardware, runs programs, manages memory and files, and gives you a user interface. Windows, macOS, Linux, Android and iOS are operating systems.

**PASS - What does a modem do for a home network?**

> A modem connects your home network to your internet service provider. It converts digital data into signals that travel over phone, cable or fibre lines, and converts them back again.

**PASS - Explain Wi-Fi to a beginner.**

> Wi-Fi is a technology that connects devices to a network without cables, using radio waves. A wireless access point, often built into your router, sends and receives the signals.

**PASS - What role does a server play?**

> A server is a computer or program that provides a service, such as a website or email. A client is a computer or program that asks the server for that service. Your web browser is a client, and the computer that sends the web page is a server.

## English generation (sampled, temperature 0.7)

**Tell me about your favourite season of the year.**

> The favourite season of the year is a wonderful time to celebrate the birth and death of the Earth. It's often celebrated with music, dancing, and festive decorations, but it's also an opportunity for outdoor activities like hiking, horseback riding, or simply taking in the sights and sounds of nature. The warm weather and sunshine can create a perfect environment for relaxation and reflection.

**Write a short story about a dog who finds a lost key.**

> As she sat hunched over her latest discovery, the dog's eyes gleamed with a mix of curiosity and desperation. It was a majestic cat, its golden tail glowing in the fading light. The owner, an elderly man named Edna, stood at the edge of his old home, his worn leather armchair bound by years of neglect. He had been searching for something - a rare, antique key that could hold the secrets to unlocking a long-lost treasure.
> 
> Edna's eyes met Emily with a mix of curiosity and trepidation. "Welcome to Edna's place, my dear friend," she said, her voice taking on a warm, gravelly voice. "I've been searching for it ever since." She gestured towards

**Why is it important to drink water every day?**

> Water is a vital resource that supports life on Earth. It's essential for human health, as it helps regulate body temperature and aids in digestion, absorption of nutrients, and waste removal. Water also helps to regulate the body's pH levels, which can affect various physiological processes, such as metabolism and immune function.
> 
> Water is also crucial for regulating the growth of cells and maintaining cellular homeostasis. It acts as a key ingredient in food sources, providing energy and reducing stress for animals and humans alike. Water also serves as an essential nutrient for human health, supporting digestion, waste removal, and waste reduction.
> 
> Additionally, water is essential for human survival and reproduction. It aids in the growth of new species, allowing them to thrive and reproduce. It also helps regulate the body's metabolism

**Can you give me some tips for studying for an exam?**

> Of course! Here are some tips for studying for an exam:
> 
> 1. **Practice Regularly**: Try to focus on your own study habits, even if it's just a few hours a day or two. This will help you stay focused and avoid burnout.
> 
> 2. **Book Study Breaks**: Look into books that can help you remember the material better. Some popular ones include "The Same as a Lunch" by Todd Hughes, "The Great Gatsby" by Julia Ballard, or "The Lost Art of Writing" by Nathalie Meyer.
> 
> 3. **Take Breaks**: Try to get your eyes on the material in front of you while studying. This can help you

**Describe what a city looks like at night.**

> The night sky appears to be a vast, ominous cloud that seems to shimmer in the sunlight. Towering palm trees cast eerie shadows on the horizon, casting a soft glow over the landscape like a thousand tiny sentinels. The air is thick with the scent of damp earth and the faint scent of decay, but there's also a hint of something more sinister lurking beneath the surface.
> 
> As the sun sets in, casting a warm orange glow over the landscape, the sky appears to be a maze of curved lines that seem to stretch out like emeralds on a hot summer day. The air is thick with smoke and dust, creating an eerie ambiance that seems almost too familiar to the outside world.
> 
> The city's buildings are

