"""Procedurally generated teaching conversations for Probabilistic Gorilla.

Three skills that a small model cannot pick up from ordinary text alone:
  * technical calculations with short worked answers (bits/bytes, binary,
    download times, subnets, unit conversions, IPv4 validity)
  * troubleshooting: symptoms -> likely cause -> what to do
  * remembering facts the user stated earlier in the conversation

Values that appear in the Gorilla benchmark (eval/gorilla_questions.json) are excluded
on purpose, so the benchmark measures whether the model generalises.
"""
import random

from handwritten import GORILLA_QA_FILES, load_qa

# ---------------------------------------------------------------- calculations
EXCLUDE = {
    ("bits_in_bytes", 6), ("bytes_in_bits", 40), ("mb_to_mbit", 25), ("gb_to_mb", 3),
    ("bin2dec", "1101"), ("bin2dec", "101010"), ("dec2bin", 10), ("subnet", 29),
    ("download", (50, 100)), ("ipv4", "192.168.1.300"),
}


def _pick(rng, options):
    return rng.choice(options)


def calc_bits_in_bytes(rng):
    n = rng.randint(1, 64)
    if ("bits_in_bytes", n) in EXCLUDE:
        return None
    q = _pick(rng, [f"How many bits are in {n} bytes?", f"How many bits make up {n} bytes?",
                    f"Convert {n} bytes to bits.", f"{n} bytes is how many bits?"])
    return q, f"Each byte has 8 bits, so {n} bytes is {n} × 8 = {n * 8} bits."


def calc_bytes_in_bits(rng):
    n = rng.randint(1, 32)
    bits = n * 8
    if ("bytes_in_bits", bits) in EXCLUDE:
        return None
    q = _pick(rng, [f"How many bytes are in {bits} bits?", f"Convert {bits} bits to bytes.",
                    f"{bits} bits is how many bytes?"])
    return q, f"There are 8 bits in a byte, so {bits} bits is {bits} ÷ 8 = {n} bytes."


def calc_mb_to_mbit(rng):
    n = rng.choice(list(range(1, 41)) + [50, 60, 75, 80, 100, 120, 150, 200, 250, 500])
    if ("mb_to_mbit", n) in EXCLUDE:
        return None
    q = _pick(rng, [f"How many megabits are in {n} megabytes?", f"A file is {n} MB. How many megabits is that?",
                    f"Convert {n} megabytes to megabits."])
    return q, f"A byte is 8 bits, so {n} megabytes is {n} × 8 = {n * 8} megabits."


def calc_gb_to_mb(rng):
    n = rng.randint(1, 20)
    if ("gb_to_mb", n) in EXCLUDE:
        return None
    unit = _pick(rng, [("gigabytes", "megabytes"), ("terabytes", "gigabytes"), ("megabytes", "kilobytes")])
    q = _pick(rng, [f"How many {unit[1]} are in {n} {unit[0]}, using 1000 per step?",
                    f"Convert {n} {unit[0]} to {unit[1]} (1 {unit[0][:-1]} = 1000 {unit[1]})."])
    return q, f"Each one of the larger unit is 1000 of the smaller, so {n} {unit[0]} is {n} × 1000 = {n * 1000} {unit[1]}."


def calc_bin2dec(rng):
    v = rng.randint(1, 255)
    b = format(v, "b")
    if ("bin2dec", b) in EXCLUDE:
        return None
    parts = [f"{int(d) * 2 ** (len(b) - 1 - i)}" for i, d in enumerate(b)]
    q = _pick(rng, [f"Convert the binary number {b} to decimal.", f"What is {b} in binary as a decimal number?",
                    f"What decimal number is binary {b}?"])
    if len(b) == 1:
        return q, f"Binary {b} is {v} in decimal."
    return q, f"Add up the place values: {' + '.join(parts)} = {v}. So binary {b} is {v} in decimal."


def calc_dec2bin(rng):
    v = rng.randint(1, 255)
    if ("dec2bin", v) in EXCLUDE:
        return None
    b = format(v, "b")
    powers = [2 ** (len(b) - 1 - i) for i, d in enumerate(b) if d == "1"]
    q = _pick(rng, [f"Write the decimal number {v} in binary.", f"What is {v} in binary?",
                    f"Convert {v} to binary."])
    return q, f"{v} = {' + '.join(map(str, powers))}, so in binary it is {b}."


def calc_download(rng):
    speed = rng.choice([8, 10, 16, 20, 40, 50, 80, 100, 200, 400, 500, 1000])
    secs = rng.randint(1, 20)
    size_mbit = speed * secs
    if size_mbit % 8:
        return None
    size = size_mbit // 8
    if ("download", (size, speed)) in EXCLUDE:
        return None
    q = _pick(rng, [f"How long does it take to download a {size} megabyte file at {speed} megabits per second?",
                    f"I have a {speed} Mbps connection. How many seconds to download {size} MB, ignoring overhead?",
                    f"Estimate the download time for {size} MB on a {speed} megabit per second link."])
    return q, (f"{size} megabytes is {size} × 8 = {size_mbit} megabits. At {speed} megabits per second that takes "
               f"{size_mbit} ÷ {speed} = {secs} seconds.")


def calc_subnet(rng):
    n = rng.randint(22, 30)
    if ("subnet", n) in EXCLUDE:
        return None
    hb = 32 - n
    q = _pick(rng, [f"How many usable host addresses are in a /{n} subnet?",
                    f"How many devices can I put in a /{n} network?",
                    f"How many hosts does a /{n} subnet allow?"])
    return q, (f"A /{n} leaves 32 − {n} = {hb} bits for hosts. 2 to the power {hb} is {2 ** hb} addresses, "
               f"minus 2 for the network and broadcast addresses, gives {2 ** hb - 2} usable hosts.")


def calc_ipv4(rng):
    parts = [rng.randint(0, 255) for _ in range(4)]
    bad = rng.random() < 0.5
    if bad:
        i = rng.randrange(4)
        parts[i] = rng.randint(256, 999)
    ip = ".".join(map(str, parts))
    if ("ipv4", ip) in EXCLUDE:
        return None
    q = _pick(rng, [f"Is {ip} a valid IPv4 address?", f"Can {ip} be an IPv4 address? Answer yes or no.",
                    f"Check this IPv4 address: {ip}"])
    if bad:
        return q, f"No. Each of the four numbers in an IPv4 address must be between 0 and 255, and {parts[i]} is too big."
    return q, "Yes. It has four numbers separated by dots, and each one is between 0 and 255."


CALCS = [calc_bits_in_bytes, calc_bytes_in_bits, calc_mb_to_mbit, calc_gb_to_mb, calc_bin2dec,
         calc_dec2bin, calc_download, calc_subnet, calc_ipv4]


# ------------------------------------------------------------- troubleshooting
DEVICES = ["laptop", "PC", "computer", "desktop", "phone", "tablet", "Mac", "work laptop"]

SCENARIOS = [
    (["My {d} can reach 1.1.1.1 with ping, but typing a website name gives an error.",
      "Pinging an IP address works from my {d}, but domain names do not resolve.",
      "Websites fail with 'server not found', yet I can reach sites by their IP address. Why?"],
     "That points to a DNS problem: your connection works, but names are not being turned into IP addresses. "
     "Try restarting the router, or set your DNS server to a public one such as 1.1.1.1 or 8.8.8.8."),
    (["My {d} shows an IP address starting with 169.254 and has no internet.",
      "Why would my {d} give itself an address like 169.254.x.x?",
      "ipconfig on my {d} shows a 169.254 address. What does that mean?"],
     "A 169.254 address means the device asked for an address with DHCP and got no answer, so it picked one itself. "
     "Check the cable or Wi-Fi connection and restart the router so its DHCP server can hand out an address."),
    (["Every device on my network lost the internet at once and the modem's status light has gone red.",
      "Every device at home lost internet at the same time, but mobile data works.",
      "All my devices say connected to Wi-Fi but no internet, and the modem light is blinking red."],
     "When every device fails at once, the problem is usually the link to your internet service provider, not your devices. "
     "Restart the modem and router, check the cables, and check whether your ISP has an outage."),
    (["My {d} says the Ethernet cable is unplugged even though a cable is connected.",
      "There is no connection on my wired {d} and the light next to the network port is off.",
      "My {d} does not detect the network cable."],
     "Check that the cable is pushed in firmly at both ends, try a different cable or a different port on the router, "
     "and make sure the network adapter is enabled."),
    (["Wi-Fi on my {d} is very slow in the bedroom but fast next to the router.",
      "My {d} keeps losing Wi-Fi when I go to the garden.",
      "Streaming buffers on my {d} upstairs, but not in the room with the router."],
     "Wi-Fi gets weaker with distance and through walls and floors, and other devices can cause interference. "
     "Move closer to the router, move the router to a central spot, or add a mesh system or access point."),
    (["My {d} gets slower and slower when I have lots of programs open.",
      "My {d} freezes when I open many apps at the same time.",
      "Task Manager shows memory usage at 98 percent and everything is sluggish."],
     "Your computer is probably running out of RAM, so it has to swap data to the much slower disk. "
     "Close programs and browser tabs you are not using, or add more RAM."),
    (["My {d} says the disk is full and it will not install updates.",
      "I cannot save files because my {d} has no storage left."],
     "Your storage is full. Delete files you no longer need, empty the recycle bin, uninstall unused apps, "
     "or move photos and videos to an external drive or cloud storage."),
    (["I got a message saying I won a prize and must enter my card details on a link.",
      "Someone emailed me pretending to be IT support and asked for my password.",
      "A text says my parcel is held and I must pay a fee through a link."],
     "That is almost certainly phishing: a fake message trying to steal your details or money. "
     "Do not click the link or reply. Delete it, and contact the company directly using its official website if you are unsure."),
    (["My {d} cannot join the Wi-Fi and says the password is wrong.",
      "My new {d} keeps failing to connect to my Wi-Fi network."],
     "Check you picked the right network name and typed the password exactly, including capital letters. "
     "If it still fails, forget the network on the device and join again, or restart the router."),
    (["A website does not load for me, but it works for my friend on another network.",
      "One particular site will not open on my {d}, but other sites are fine."],
     "The problem is probably on your side. Try clearing the browser cache, using another browser, "
     "restarting your router, or changing your DNS server. A firewall or VPN can also block a site."),
    (["My {d} fan is loud and it shuts down by itself when I play games.",
      "My {d} gets very hot and turns off suddenly."],
     "It is probably overheating. Make sure the air vents are not blocked, clean out dust, use it on a hard flat surface, "
     "and check that the fans are working."),
    (["My {d} shows lots of pop-up adverts and new toolbars I never installed.",
      "Programs I did not install keep appearing on my {d}."],
     "That sounds like malware. Run a full scan with your antivirus, remove programs you do not recognise, "
     "update your system and change your passwords from a clean device."),
    (["I can reach my web server inside the office, but people outside cannot connect to it.",
      "My game server works on my LAN but my friends on the internet cannot join."],
     "The router or firewall is probably blocking incoming connections. Set up port forwarding on the router to "
     "the server's local IP address and allow the port in the firewall."),
    (["My video calls stutter even though my download speed test looks fine.",
      "Online games lag on my {d} although my bandwidth is high."],
     "Real-time apps care most about latency and packet loss, not just bandwidth. Use a wired Ethernet connection, "
     "stop large downloads while calling, and restart the router."),
    (["My printer shows as offline on my {d}.",
      "My {d} cannot find the wireless printer any more."],
     "Make sure the printer is switched on and connected to the same network as your computer. "
     "Restart the printer and the computer, and re-add the printer if its IP address has changed."),
]


def troubleshoot(rng):
    symptoms, answer = rng.choice(SCENARIOS)
    q = rng.choice(symptoms).replace("{d}", rng.choice(DEVICES))
    if rng.random() < 0.3:
        q += " " + rng.choice(["What should I do?", "Any ideas?", "How can I fix it?", "What could be wrong?"])
    return q, answer


# --------------------------------------------------------------------- memory
NAMES = ["Sam", "Alex", "Maria", "Tom", "Aisha", "Leo", "Chen", "Fatima", "Oliver", "Mei", "Jonas", "Zara",
         "Diego", "Hannah", "Kofi", "Ines", "Ravi", "Sofia", "Ben", "Yuki"]
WORDS = ["Blue", "Silver", "Happy", "Quiet", "Green", "Rapid", "Lucky", "Purple", "Golden", "Sleepy"]
ANIMALS = ["Fox", "Owl", "Tiger", "Panda", "Otter", "Falcon", "Koala", "Wolf", "Lynx", "Moose"]
ITEMS = ["file server", "printer", "NAS", "web server", "game server", "camera", "smart TV", "database server"]
EXCLUDE_VALUES = {"priya", "orangebadger", "10.20.30.40", "192.168.50.1"}


def _ip(rng):
    return rng.choice([f"192.168.{rng.randint(0, 99)}.{rng.randint(1, 254)}",
                       f"10.{rng.randint(0, 99)}.{rng.randint(0, 99)}.{rng.randint(1, 254)}"])


def _fact(rng):
    kind = rng.randrange(6)
    if kind == 0:
        v = rng.choice(NAMES)
        return (f"Hi, I'm {v}.", f"Nice to meet you, {v}! How can I help?", "What's my name?",
                f"Your name is {v}.", v)
    if kind == 1:
        item, v = rng.choice(ITEMS), _ip(rng)
        return (f"My {item} has the IP address {v}.", f"Got it, your {item} is at {v}.",
                rng.choice([f"What IP address did I give for my {item}?", f"What was my {item}'s IP address?"]),
                f"You said your {item} has the IP address {v}.", v)
    if kind == 2:
        v = rng.choice([4, 8, 12, 16, 24, 32, 64])
        return (f"My computer has {v} GB of RAM.", f"Thanks, {v} GB of RAM noted.",
                rng.choice(["How much RAM does my computer have?", "How much memory did I say I have?"]),
                f"You said your computer has {v} GB of RAM.", str(v))
    if kind == 3:
        v = rng.choice(WORDS) + rng.choice(ANIMALS)
        return (f"My home Wi-Fi network is named {v}.", f"OK, your Wi-Fi network is {v}.",
                rng.choice(["What is my Wi-Fi called?", "What did I say my network name was?"]),
                f"Your Wi-Fi network is called {v}.", v)
    if kind == 4:
        v = _ip(rng)
        return (f"The admin page of my router is {v}.", f"Understood, the router's admin page is at {v}.",
                rng.choice(["Where is my router's admin page again?", "What address did I give for the router admin page?"]),
                f"You said the router's admin page is at {v}.", v)
    v = rng.choice(["Linux", "Windows", "macOS", "ChromeOS"])
    return (f"I use {v} on my main computer.", f"Good to know, you use {v}.",
            rng.choice(["Which operating system did I say I use?", "What OS is on my main computer?"]),
            f"You said you use {v}.", v)


def memory_conversation(rng, qa_blocks):
    for _ in range(10):
        say, ack, ask, answer, value = _fact(rng)
        if value.lower() not in EXCLUDE_VALUES:
            break
    conv = [{"role": "user", "content": say}, {"role": "assistant", "content": ack}]
    for qs, a in rng.sample(qa_blocks, rng.randint(1, 3)):
        conv += [{"role": "user", "content": rng.choice(qs)}, {"role": "assistant", "content": a}]
    conv += [{"role": "user", "content": ask}, {"role": "assistant", "content": answer}]
    return conv


# ---------------------------------------------------------------------- build
def conversations(n, seed=0):
    """n synthetic conversations mixing calculations, troubleshooting and memory."""
    rng = random.Random(seed)
    qa_blocks = load_qa(GORILLA_QA_FILES, identity="gorilla")
    out = []
    while len(out) < n:
        r = rng.random()
        if r < 0.45:
            item = rng.choice(CALCS)(rng)
        elif r < 0.75:
            item = troubleshoot(rng)
        else:
            out.append(memory_conversation(rng, qa_blocks))
            continue
        if item:
            out.append([{"role": "user", "content": item[0]}, {"role": "assistant", "content": item[1]}])
    return out


if __name__ == "__main__":
    from common import format_chat
    for c in conversations(8, seed=1):
        print(format_chat(c), "\n")
