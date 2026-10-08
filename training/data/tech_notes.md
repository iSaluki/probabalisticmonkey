# How Computers Work

A computer is a machine that follows instructions to process information. It takes input, processes it, stores it, and produces output. Every computer, from a phone to a large server, is built from the same basic parts: a processor, memory, storage, and input and output devices.

## The CPU

The central processing unit, or CPU, is the part of a computer that carries out instructions. It is often called the brain of the computer. The CPU fetches an instruction from memory, decodes it to work out what it means, and then executes it. This is called the fetch, decode, execute cycle, and a modern CPU repeats it billions of times every second. The speed of a CPU is measured in hertz; a 3 gigahertz processor runs three billion clock cycles per second. Most modern CPUs have several cores, and each core can work on its own task at the same time as the others.

## Memory (RAM)

Random access memory, or RAM, is the short-term working memory of a computer. The CPU keeps the programs and data it is using right now in RAM because RAM is very fast. RAM is volatile, which means it loses everything when the power is turned off. Having more RAM lets a computer run more programs at once without slowing down.

## Storage

Storage keeps data even when the computer is switched off. A hard disk drive (HDD) stores data on spinning magnetic platters. A solid state drive (SSD) stores data in flash memory chips and has no moving parts, so it is much faster than a hard disk. Storage is slower than RAM but it is permanent and much larger. Files, programs and the operating system are all kept in storage.

## Input and output

Input devices send information into a computer. Examples are the keyboard, the mouse, a touch screen, a microphone and a camera. Output devices show or send information out of the computer. Examples are the monitor, speakers and a printer. Some devices, like a network card or a USB drive, do both input and output.

## The motherboard and GPU

The motherboard is the main circuit board of a computer. It connects the CPU, the memory, the storage and every other part so that they can talk to each other. The graphics processing unit, or GPU, is a processor designed to do many simple calculations at the same time. It draws images on the screen, and it is also used for games, video editing and training machine learning models.

## Hardware and software

Hardware is the physical parts of a computer that you can touch, such as the keyboard, the CPU and the screen. Software is the set of programs and instructions that tell the hardware what to do. A web browser, a game and a word processor are all software. Hardware without software cannot do anything useful, and software needs hardware to run on.

## The operating system

An operating system (OS) is the main software that manages a computer. It controls the hardware, runs programs, manages memory and files, and provides a user interface. Windows, macOS, Linux, Android and iOS are operating systems. When you open an app, the operating system loads it into memory and gives it time on the CPU.

## Binary, bits and bytes

Computers store and process all information as binary numbers, using only the digits 0 and 1. A single binary digit is called a bit. Eight bits make one byte, and one byte can store a number from 0 to 255 or a single letter of text. A kilobyte is about a thousand bytes, a megabyte is about a million bytes, a gigabyte is about a billion bytes and a terabyte is about a trillion bytes. Computers use binary because their circuits have two states: on and off.

## Programs and programming languages

A program is a list of instructions that a computer follows to do a task. Programmers write programs in programming languages such as Python, JavaScript, C and Java. A compiler translates a whole program into machine code before it runs, while an interpreter translates and runs a program one line at a time. An algorithm is a step by step method for solving a problem, like a recipe that a computer can follow.

## Files and folders

A file is a named collection of data stored on a disk, such as a document, a photo or a program. Files are organised into folders, which are also called directories. The file system is the part of the operating system that keeps track of where every file is stored.

# How Networks Work

A computer network is a group of computers and other devices connected together so they can share data and resources. Devices on a network can send messages, share files, share printers and connect to the internet. Networks can be wired, using cables, or wireless, using radio signals.

## LAN and WAN

A local area network, or LAN, connects devices in a small area such as a home, a school or an office. A wide area network, or WAN, connects networks across a large area such as a country or the whole world. The internet is the largest wide area network.

## The internet

The internet is a global network of networks. It connects billions of computers, phones and servers around the world. Data travels across the internet in small pieces called packets, and every device on the internet follows shared rules called protocols so that they can understand each other. The World Wide Web is a service that runs on the internet; it is the collection of web pages that you visit with a browser.

## Packets

When data is sent over a network, it is broken into small pieces called packets. Each packet carries part of the data plus a header with the source address and the destination address. Packets can travel by different routes and are put back together in the right order when they arrive. Sending data in packets makes networks efficient and reliable, because a lost packet can simply be sent again.

## Protocols

A protocol is a set of rules that devices follow to communicate on a network. Protocols define how data is formatted, addressed, sent and received. Important protocols include IP, TCP, UDP, HTTP, HTTPS, DNS and DHCP.

## IP addresses

An IP address is a unique number that identifies a device on a network, much like a postal address identifies a house. An IPv4 address is made of four numbers from 0 to 255 separated by dots, for example 192.168.1.10. IPv4 has about four billion addresses, which is not enough for every device in the world, so the newer IPv6 uses much longer addresses written in hexadecimal, such as 2001:db8::1. Private addresses like 192.168.x.x are used inside home networks.

## DNS

The Domain Name System, or DNS, translates human friendly domain names into IP addresses. When you type a name such as example.com into a browser, your computer asks a DNS server for the IP address of that name, and then it connects to that address. DNS is often called the phone book of the internet.

## Routers

A router is a device that forwards packets between different networks. It reads the destination IP address of each packet and chooses the best path to send it on. Your home router connects your local network to your internet service provider, and routers across the internet pass packets along until they reach their destination.

## Switches and hubs

A switch connects devices within the same local network. It learns the MAC address of each connected device and sends each frame only to the port where the destination device is. A hub is an older and simpler device that sends every message to all of its ports.

## Modems

A modem connects a home or office network to an internet service provider. It converts digital data from your computer into signals that can travel over a phone line, cable line or fibre connection, and converts incoming signals back into digital data. The name comes from modulator-demodulator.

## MAC addresses

A MAC address is a hardware address built into a network interface card. It identifies a device on a local network and is usually written as six pairs of hexadecimal digits, such as 00:1A:2B:3C:4D:5E. IP addresses are used to route data between networks, while MAC addresses deliver data within a local network.

## TCP and UDP

TCP, the Transmission Control Protocol, provides reliable delivery of data. It sets up a connection with a three way handshake, numbers every packet, checks that packets arrive, resends lost packets and puts them back in order. TCP is used for web pages, email and file transfers. UDP, the User Datagram Protocol, is faster and simpler but does not guarantee delivery or order. UDP is used for live video, online games, voice calls and DNS lookups, where speed matters more than perfect reliability.

## Ports

A port is a number that identifies a specific service or program on a computer. The IP address finds the computer, and the port finds the right program on it. For example, HTTP uses port 80, HTTPS uses port 443, SSH uses port 22 and DNS uses port 53.

## HTTP and HTTPS

HTTP, the Hypertext Transfer Protocol, is the protocol that web browsers and web servers use to exchange web pages. A browser sends an HTTP request, such as GET, and the server sends back a response with a status code, such as 200 for OK or 404 for not found. HTTPS is the secure version of HTTP. It encrypts the data with TLS so that other people cannot read or change it, which is why you see a padlock in the browser.

## Clients and servers

A server is a computer or program that provides a service, such as a website, email or files, to other computers. A client is a computer or program that asks a server for that service. When you visit a website, your web browser is the client and the computer that sends you the page is the server.

## URLs and web browsers

A URL, or uniform resource locator, is the address of a resource on the web, such as https://www.example.com/index.html. It includes the protocol, the domain name and the path. A web browser is a program that requests web pages from servers and displays them. Chrome, Firefox, Safari and Edge are web browsers.

## DHCP

DHCP, the Dynamic Host Configuration Protocol, automatically gives each device that joins a network an IP address and other settings, such as the default gateway and the DNS server. Without DHCP, every address would have to be set by hand.

## NAT and gateways

The default gateway is the router that a device sends traffic to when the destination is outside its own local network. Network address translation, or NAT, lets many devices in a home share one public IP address. The router replaces the private addresses of outgoing packets with its public address and keeps track of which reply belongs to which device.

## Wi-Fi and Ethernet

Wi-Fi is a technology that connects devices to a network wirelessly using radio waves. Ethernet is a wired networking technology that uses cables and is usually faster and more stable than Wi-Fi. Most homes use a wireless access point, often built into the router, so that laptops and phones can join the network.

## Bandwidth and latency

Bandwidth is the maximum amount of data a connection can carry each second, measured in bits per second, such as 100 megabits per second. Latency is the delay for data to travel from one point to another, measured in milliseconds. A connection can have high bandwidth but also high latency. The ping command measures latency by sending a small message to another computer and timing the reply.

## Firewalls

A firewall is a security system that monitors network traffic and blocks or allows it according to a set of rules. It acts like a barrier between a trusted network and an untrusted network such as the internet, and it helps stop attackers and unwanted connections.

## Encryption

Encryption scrambles data so that only someone with the right key can read it. It protects passwords, messages and payments as they travel across a network. HTTPS, VPNs and secure messaging apps all use encryption.

## VPNs

A virtual private network, or VPN, creates an encrypted tunnel between your device and another network over the internet. It protects your traffic on public Wi-Fi and lets workers connect securely to their office network.

## The OSI and TCP/IP models

The OSI model describes networking in seven layers: physical, data link, network, transport, session, presentation and application. The simpler TCP/IP model has four layers: link, internet, transport and application. IP works at the network or internet layer, TCP and UDP work at the transport layer, and HTTP and DNS work at the application layer.

## Email

Email is sent across the internet using protocols such as SMTP, the Simple Mail Transfer Protocol, which sends messages between mail servers. IMAP and POP3 are protocols used to download or read email from a server.

## The cloud

Cloud computing means using computers, storage and software that are run by another company over the internet instead of on your own machine. Online file storage, web email and streaming services all run in the cloud, inside large buildings full of servers called data centres.

## Staying safe online

Malware is harmful software, such as viruses, worms and ransomware. Phishing is a trick where an attacker sends a fake message to steal passwords or money. You can stay safe by using strong and unique passwords, turning on two factor authentication, keeping software updated and not clicking suspicious links.
