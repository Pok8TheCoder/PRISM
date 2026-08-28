# PRISM — Complete Feature Documentation

> **Predictive Recurrent Infiltration State Model (PRISM)**
> Comprehensive listing of all 374 raw, engineered, packet, and aggregated state vector features.

---

## 📊 Summary Overview

| Category / Component | Raw Features | Processed & Engineered Features |
|---|:---:|:---:|
| **1. CSE-CIC-IDS2018 & CTU-13** | 78 | 109 |
| **2. Packet-Level PCAP Parser** | 12 | 12 |
| **3. UNSW-NB15 Dataset** | 49 | 76 |
| **4. CICIoT2023 Dataset** | 46 | 73 |
| **5. LANL Authentication Dataset** | 9 | 36 |
| **6. DARPA Intrusion Detection Dataset** | 41 | 68 |
| **Total Unique Extracted Features** | **235** | **374** |

---

## 1. 🌊 CIC-IDS-2018 & CTU-13 Flow Features (109 Total)

### Raw Netflow Features (78 Features)
1. `Dst Port` — Destination Port number
2. `Protocol` — IP Protocol number (6=TCP, 17=UDP, 1=ICMP)
3. `Timestamp` — Connection timestamp
4. `Flow Duration` — Total duration of flow in microseconds
5. `Tot Fwd Pkts` — Total packets in forward direction
6. `Tot Bwd Pkts` — Total packets in backward direction
7. `TotLen Fwd Pkts` — Total size of packets in forward direction (bytes)
8. `TotLen Bwd Pkts` — Total size of packets in backward direction (bytes)
9. `Fwd Pkt Len Max` — Maximum packet length in forward direction
10. `Fwd Pkt Len Min` — Minimum packet length in forward direction
11. `Fwd Pkt Len Mean` — Mean packet length in forward direction
12. `Fwd Pkt Len Std` — Standard deviation of packet length in forward direction
13. `Bwd Pkt Len Max` — Maximum packet length in backward direction
14. `Bwd Pkt Len Min` — Minimum packet length in backward direction
15. `Bwd Pkt Len Mean` — Mean packet length in backward direction
16. `Bwd Pkt Len Std` — Standard deviation of packet length in backward direction
17. `Flow Byts/s` — Flow throughput in bytes per second
18. `Flow Pkts/s` — Flow rate in packets per second
19. `Flow IAT Mean` — Mean inter-arrival time between packets in flow
20. `Flow IAT Std` — Standard deviation of inter-arrival time
21. `Flow IAT Max` — Maximum inter-arrival time
22. `Flow IAT Min` — Minimum inter-arrival time
23. `Fwd IAT Tot` — Total inter-arrival time in forward direction
24. `Fwd IAT Mean` — Mean inter-arrival time in forward direction
25. `Fwd IAT Std` — Standard deviation of inter-arrival time in forward direction
26. `Fwd IAT Max` — Maximum inter-arrival time in forward direction
27. `Fwd IAT Min` — Minimum inter-arrival time in forward direction
28. `Bwd IAT Tot` — Total inter-arrival time in backward direction
29. `Bwd IAT Mean` — Mean inter-arrival time in backward direction
30. `Bwd IAT Std` — Standard deviation of inter-arrival time in backward direction
31. `Bwd IAT Max` — Maximum inter-arrival time in backward direction
32. `Bwd IAT Min` — Minimum inter-arrival time in backward direction
33. `Fwd PSH Flags` — Number of times PSH flag was set in forward direction
34. `Bwd PSH Flags` — Number of times PSH flag was set in backward direction
35. `Fwd URG Flags` — Number of times URG flag was set in forward direction
36. `Bwd URG Flags` — Number of times URG flag was set in backward direction
37. `Fwd Header Len` — Total header bytes in forward direction
38. `Bwd Header Len` — Total header bytes in backward direction
39. `Fwd Pkts/s` — Forward packets per second
40. `Bwd Pkts/s` — Backward packets per second
41. `Pkt Len Min` — Minimum packet length across flow
42. `Pkt Len Max` — Maximum packet length across flow
43. `Pkt Len Mean` — Mean packet length across flow
44. `Pkt Len Std` — Standard deviation of packet length
45. `Pkt Len Var` — Variance of packet length
46. `FIN Flag Cnt` — Number of packets with FIN flag
47. `SYN Flag Cnt` — Number of packets with SYN flag
48. `RST Flag Cnt` — Number of packets with RST flag
49. `PSH Flag Cnt` — Number of packets with PSH flag
50. `ACK Flag Cnt` — Number of packets with ACK flag
51. `URG Flag Cnt` — Number of packets with URG flag
52. `CWE Flag Count` — Number of packets with CWE flag
53. `ECE Flag Cnt` — Number of packets with ECE flag
54. `Down/Up Ratio` — Download to upload ratio
55. `Pkt Size Avg` — Average payload size
56. `Fwd Seg Size Avg` — Average segment size in forward direction
57. `Bwd Seg Size Avg` — Average segment size in backward direction
58. `Fwd Byts/b Avg` — Average forward bytes per bulk
59. `Fwd Pkts/b Avg` — Average forward packets per bulk
60. `Fwd Blk Rate Avg` — Average forward bulk rate
61. `Bwd Byts/b Avg` — Average backward bytes per bulk
62. `Bwd Pkts/b Avg` — Average backward packets per bulk
63. `Bwd Blk Rate Avg` — Average backward bulk rate
64. `Subflow Fwd Pkts` — Average forward subflow packet count
65. `Subflow Fwd Byts` — Average forward subflow byte count
66. `Subflow Bwd Pkts` — Average backward subflow packet count
67. `Subflow Bwd Byts` — Average backward subflow byte count
68. `Init Fwd Win Byts` — Initial TCP window bytes in forward direction
69. `Init Bwd Win Byts` — Initial TCP window bytes in backward direction
70. `Fwd Act Data Pkts` — Packets with minimum 1 byte payload in forward direction
71. `Fwd Seg Size Min` — Minimum segment size in forward direction
72. `Active Mean` — Mean active time before idling
73. `Active Std` — Standard deviation of active time
74. `Active Max` — Maximum active time
75. `Active Min` — Minimum active time
76. `Idle Mean` — Mean idle time before becoming active
77. `Idle Std` — Standard deviation of idle time
78. `Idle Max` — Maximum idle time

### Engineered Flow Features (31 Features)
79. `fwd_bwd_ratio` — `Tot Fwd Pkts / (Tot Bwd Pkts + 1e-6)`
80. `bytes_per_packet` — `(TotLen Fwd + TotLen Bwd) / (Tot Fwd + Tot Bwd)`
81. `flag_syn_frac` — Proportion of SYN flags to total packets
82. `flag_ack_frac` — Proportion of ACK flags to total packets
83. `flag_fin_frac` — Proportion of FIN flags to total packets
84. `flag_rst_frac` — Proportion of RST flags to total packets
85. `flag_psh_frac` — Proportion of PSH flags to total packets
86. `flag_urg_frac` — Proportion of URG flags to total packets
87. `port_cat_well_known` — Flag for port in range 0–1023
88. `port_cat_registered` — Flag for port in range 1024–49151
89. `port_cat_ephemeral` — Flag for port in range 49152–65535
90. `port_21` — Binary indicator for FTP port 21
91. `port_22` — Binary indicator for SSH port 22
92. `port_23` — Binary indicator for Telnet port 23
93. `port_25` — Binary indicator for SMTP port 25
94. `port_53` — Binary indicator for DNS port 53
95. `port_80` — Binary indicator for HTTP port 80
96. `port_110` — Binary indicator for POP3 port 110
97. `port_135` — Binary indicator for RPC port 135
98. `port_139` — Binary indicator for NetBIOS port 139
99. `port_143` — Binary indicator for IMAP port 143
100. `port_443` — Binary indicator for HTTPS port 443
101. `port_445` — Binary indicator for SMB port 445
102. `port_993` — Binary indicator for IMAPS port 993
103. `port_995` — Binary indicator for POP3S port 995
104. `port_1433` — Binary indicator for MSSQL port 1433
105. `port_1723` — Binary indicator for PPTP port 1723
106. `port_3306` — Binary indicator for MySQL port 3306
107. `port_3389` — Binary indicator for RDP port 3389
108. `port_5900` — Binary indicator for VNC port 5900
109. `port_8080` — Binary indicator for HTTP-Proxy port 8080

---

## 2. 📦 Packet-Level PCAP Features (12 Total)
110. `ttl_mean` — Mean Time-To-Live across packets
111. `ttl_std` — Standard deviation of Time-To-Live
112. `tcp_window_mean` — Mean TCP advertised window size
113. `tcp_window_std` — Standard deviation of TCP window size
114. `ip_frag_flag` — Count of fragmented IP packets
115. `payload_size_mean` — Mean application payload byte size
116. `payload_size_std` — Standard deviation of payload byte size
117. `port_scan_sequential` — Counter for sequential port probing sequence
118. `port_scan_random` — Counter for random high-entropy port probing
119. `retransmission_count` — Counter for TCP retransmissions
120. `syn_ack_ratio` — Ratio of SYN packets to ACK packets (half-open probe indicator)
121. `rst_rate` — Abrupt connection reset rate

---

## 3. 🇦🇺 UNSW-NB15 Features (76 Total)

### Raw Features (49 Features)
122. `srcip` — Source IP address
123. `sport` — Source Port number
124. `dstip` — Destination IP address
125. `dsport` — Destination Port number
126. `proto` — Transaction Protocol (tcp, udp, icmp, etc.)
127. `state` — Transaction State (FIN, INT, CON, REJ, etc.)
128. `dur` — Record duration
129. `sbytes` — Source to destination transaction bytes
130. `dbytes` — Destination to source transaction bytes
131. `sttl` — Source Time-To-Live value
132. `dttl` — Destination Time-To-Live value
133. `sloss` — Source packets retransmitted or dropped
134. `dloss` — Destination packets retransmitted or dropped
135. `service` — Application service (http, dns, ftp, smtp, ssh, etc.)
136. `sload` — Source bits per second
137. `dload` — Destination bits per second
138. `spkts` — Source to destination packet count
139. `dpkts` — Destination to source packet count
140. `swin` — Source TCP window advertisement
141. `dwin` — Destination TCP window advertisement
142. `stcpb` — Source TCP sequence number
143. `dtcpb` — Destination TCP sequence number
144. `smean` — Mean packet size transmitted by source
145. `dmean` — Mean packet size transmitted by destination
146. `trans_depth` — Pipelined HTTP request depth
147. `response_body_len` — Content body length from HTTP response
148. `sjit` — Source inter-packet jitter (ms)
149. `djit` — Destination inter-packet jitter (ms)
150. `stime` — Record start time
151. `ltime` — Record last packet time
152. `sintpkt` — Source inter-packet arrival time
153. `dintpkt` — Destination inter-packet arrival time
154. `tcprtt` — TCP Round Trip Time (`synack` + `ackdat`)
155. `synack` — Time between SYN and SYN-ACK packet
156. `ackdat` — Time between SYN-ACK and ACK packet
157. `is_sm_ips_ports` — Flag for identical src/dst IP and port
158. `ct_state_ttl` — Count of connections sharing state and TTL
159. `ct_flw_http_mthd` — Count of HTTP methods (GET/POST) in flow
160. `is_ftp_login` — Binary flag for FTP authentication attempt
161. `ct_ftp_cmd` — Count of FTP session command flows
162. `ct_srv_src` — Count of connections with same service & src IP
163. `ct_srv_dst` — Count of connections with same service & dst IP
164. `ct_dst_ltm` — Connections to same dst IP in recent time
165. `ct_src_ltm` — Connections from same src IP in recent time
166. `ct_src_dport_ltm` — Connections from src IP to dst port in recent time
167. `ct_dst_sport_ltm` — Connections to dst IP from src port in recent time
168. `ct_dst_src_ltm` — Connections between src & dst IP in recent time
169. `attack_cat` — Ground truth attack category string
170. `label` — Binary attack label (0=Normal, 1=Attack)

### Engineered Features (27 Features)
171. `fwd_bwd_ratio` — `spkts / (dpkts + 1e-6)`
172. `port_cat_well_known` — Destination port in range 0–1023
173. `port_cat_registered` — Destination port in range 1024–49151
174. `port_cat_ephemeral` — Destination port in range 49152–65535
175. `flag_syn_frac` — Fraction of SYN flags
176. `flag_ack_frac` — Fraction of ACK flags
177. `flag_fin_frac` — Fraction of FIN flags
178. `flag_rst_frac` — Fraction of RST flags
179. `flag_psh_frac` — Fraction of PSH flags
180. `flag_urg_frac` — Fraction of URG flags
181–197. `port_{21,22,23,25,53,80,110,135,139,143,443,445,993,995,1433,1723,3306,3389,5900,8080}` — Top port indicators

---

## 4. 🌐 CICIoT2023 IoT Features (73 Total)

### Raw Features (46 Features)
198. `flow_duration` — Flow duration in seconds
199. `Header_Length` — Total header length
200. `Protocol Type` — Protocol type ID
201. `Duration` — Transaction duration
202. `Rate` — Total transfer rate (bytes/s)
203. `Srate` — Source packet send rate
204. `Drate` — Destination packet receive rate
205. `fin_flag_number` — Binary FIN flag presence
206. `syn_flag_number` — Binary SYN flag presence
207. `rst_flag_number` — Binary RST flag presence
208. `psh_flag_number` — Binary PSH flag presence
209. `ack_flag_number` — Binary ACK flag presence
210. `ece_flag_number` — Binary ECE flag presence
211. `cwr_flag_number` — Binary CWR flag presence
212. `ack_count` — Total ACK packet count
213. `syn_count` — Total SYN packet count
214. `fin_count` — Total FIN packet count
215. `urg_count` — Total URG packet count
216. `rst_count` — Total RST packet count
217. `HTTP` — Binary HTTP application protocol flag
218. `HTTPS` — Binary HTTPS application protocol flag
219. `DNS` — Binary DNS application protocol flag
220. `Telnet` — Binary Telnet application protocol flag
221. `SMTP` — Binary SMTP application protocol flag
222. `SSH` — Binary SSH application protocol flag
223. `IRC` — Binary IRC application protocol flag
224. `TCP` — Binary TCP transport protocol flag
225. `UDP` — Binary UDP transport protocol flag
226. `DHCP` — Binary DHCP protocol flag
227. `ARP` — Binary ARP link layer flag
228. `ICMP` — Binary ICMP protocol flag
229. `IPv` — Binary IP protocol flag
230. `LLC` — Binary Logical Link Control protocol flag
231. `Tot sum` — Total bytes transferred across flow
232. `Min` — Minimum packet size
233. `Max` — Maximum packet size
234. `AVG` — Average packet size
235. `Std` — Standard deviation of packet size
236. `Tot size` — Total payload size
237. `IAT` — Inter-arrival time mean
238. `Number` — Number of packets in flow
239. `Magnitue` — Vector magnitude of packet rates
240. `Radius` — Dispersion radius of packet lengths
241. `Covariance` — Covariance of rate vs size
242. `Variance` — Variance of packet arrival rates
243. `Weight` — Weight assigned to packet bursts

### Engineered Features (27 Features)
244–249. `flag_{syn,ack,fin,rst,psh,urg}_frac` — TCP flag proportions
250–252. `port_cat_{well_known,registered,ephemeral}` — Port range flags
253–270. `port_{21,22,23,25,53,80,110,135,139,143,443,445,993,995,1433,1723,3306,3389,5900,8080}` — Top 20 port indicators

---

## 5. 🔑 LANL Authentication Dataset Features (36 Total)

### Raw Fields (9 Features)
271. `time` — Epoch timestamp of auth event
272. `src_user` — Source user identifier (`U123@DOM1`)
273. `dst_user` — Destination user identifier
274. `src_comp` — Source computer hostname (`C456`)
275. `dst_comp` — Destination computer hostname
276. `auth_type` — Authentication protocol (Kerberos, NTLM, Negotiate)
277. `logon_type` — Logon mode (Interactive, Network, Service, Batch)
278. `orientation` — Token operation (LogOn, LogOff, TGT, TGS)
279. `result` — Outcome string (Success / Fail)

### Engineered Flow-Mapped Features (27 Features)
280. `Flow Duration` — Fixed 1.0 (discrete event indicator)
281. `Tot Fwd Pkts` — Fixed 1
282. `Tot Bwd Pkts` — Fixed 1
283. `TotLen Fwd Pkts` — Fixed 64 bytes
284. `TotLen Bwd Pkts` — Fixed 64 bytes
285. `Dst Port` — Inferred port (88 for Kerberos, 445 for SMB)
286. `Protocol` — Fixed 6 (TCP)
287. `Flow Byts/s` — Calculated 128 B/s
288. `Flow Pkts/s` — Calculated 2 Pkts/s
289. `fwd_bwd_ratio` — Fixed 1.0
290–292. `port_cat_{well_known,registered,ephemeral}` — Port category indicators
293–298. `flag_{syn,ack,fin,rst,psh,urg}_frac` — Zero-filled flag placeholders
299–306. `port_{21,22,23,25,53,80,110,135,139,143,443,445,993,995,1433,1723,3306,3389,5900,8080}` — Port indicators

---

## 6. 🏛️ DARPA Intrusion Detection Dataset Features (68 Total)

### Raw Features (41 Features)
307. `duration` — Connection duration in seconds
308. `protocol_type` — Protocol string (tcp, udp, icmp)
309. `service` — Network service (http, smtp, ftp, domain_u, etc.)
310. `flag` — Connection status flag (SF, S0, REJ, RSTO, RSTR, etc.)
311. `src_bytes` — Bytes sent from source to destination
312. `dst_bytes` — Bytes sent from destination to source
313. `land` — Binary flag (1 if src and dst IP/port match)
314. `wrong_fragment` — Count of corrupted/wrong fragments
315. `urgent` — Count of urgent packets
316. `hot` — Count of hot indicators (e.g. access to system files)
317. `num_failed_logins` — Count of failed authentication attempts
318. `logged_in` — Binary flag (1 if successfully logged in)
319. `num_compromised` — Count of compromised conditions
320. `root_shell` — Binary flag (1 if root shell obtained)
321. `su_attempted` — Binary flag (1 if `su root` attempted)
322. `num_root` — Count of root accesses
323. `num_file_creations` — Count of file creation operations
324. `num_shells` — Count of shell prompts opened
325. `num_access_files` — Count of operations on control files
326. `is_host_login` — Binary flag (1 if login is host login)
327. `is_guest_login` — Binary flag (1 if guest login)
328. `count` — Connections to same host as current connection in last 2 sec
329. `srv_count` — Connections to same service in last 2 sec
330. `serror_rate` — % of connections that activated SYN error flags
331. `srv_serror_rate` — % of service connections with SYN error flags
332. `rerror_rate` — % of connections with REJ error flags
333. `srv_rerror_rate` — % of service connections with REJ error flags
334. `same_srv_rate` — % of connections to same service
335. `diff_srv_rate` — % of connections to different services
336. `srv_diff_host_rate` — % of connections to different hosts
337. `dst_host_count` — Count of destination host connections
338. `dst_host_srv_count` — Count of destination host service connections
339. `dst_host_same_srv_rate` — % of dest host connections to same service
340. `dst_host_diff_srv_rate` — % of dest host connections to diff service
341. `dst_host_same_src_port_rate` — % of dest host connections from same src port
342. `dst_host_srv_diff_host_rate` — % of dest host service connections from diff host
343. `dst_host_serror_rate` — % of dest host connections with SYN errors
344. `dst_host_srv_serror_rate` — % of dest host service connections with SYN errors
345. `dst_host_rerror_rate` — % of dest host connections with REJ errors
346. `dst_host_srv_rerror_rate` — % of dest host service connections with REJ errors
347. `label` — Target attack class (normal, probe, dos, u2r, r2l)

### Engineered Features (27 Features)
348. `Tot Fwd Pkts` — Estimated `ceil(src_bytes / 500)`
349. `Tot Bwd Pkts` — Estimated `ceil(dst_bytes / 500)`
350. `Flow Byts/s` — `(src_bytes + dst_bytes) / (duration + 1e-6)`
351. `Flow Pkts/s` — `(Tot Fwd Pkts + Tot Bwd Pkts) / (duration + 1e-6)`
352. `fwd_bwd_ratio` — `Tot Fwd Pkts / (Tot Bwd Pkts + 1e-6)`
353–355. `port_cat_{well_known,registered,ephemeral}` — Port range indicators
356–361. `flag_{syn,ack,fin,rst,psh,urg}_frac` — TCP flag proportions
362–374. `port_{21,22,23,25,53,80,110,135,139,143,443,445,993,995,1433,1723,3306,3389,5900,8080}` — Port indicators

---

## 7. 🧮 Aggregated State Vector Features ($S_t$, ~110 Features)

Constructed per 30-second time window for model ingestion:

1. `num_flows` — Total flow records aggregated in window
2. `num_unique_src_ips` — Count of distinct source IP addresses
3. `num_unique_dst_ips` — Count of distinct destination IP addresses
4. `num_unique_dst_ports` — Count of distinct destination ports
5. `port_entropy` — Shannon entropy calculated over destination port distribution
6–85. `mean_*` and `std_*` for all numeric feature columns (Flow Duration, Bytes/s, Packet lengths, IAT, etc.)
86. `flag_syn_frac` — Window average SYN flag fraction
87. `flag_ack_frac` — Window average ACK flag fraction
88. `flag_fin_frac` — Window average FIN flag fraction
89. `flag_rst_frac` — Window average RST flag fraction
90. `flag_psh_frac` — Window average PSH flag fraction
91. `flag_urg_frac` — Window average URG flag fraction
92. `proto_frac_6` — Proportion of TCP flows in window
93. `proto_frac_17` — Proportion of UDP flows in window
94. `proto_frac_1` — Proportion of ICMP flows in window
95. `proto_frac_other` — Proportion of other protocol flows
96–105. `top_port_{22,80,443,445,3389,21,53,135,139,1433}_count` — Flow counts targeting top 10 ports
106. `has_packet_features` — Binary indicator (1 if PCAP features present, 0 if CSV-only)
107. `ttl_mean` — Mean packet TTL across window
108. `tcp_window_mean` — Mean TCP window size across window
109. `retransmission_count` — Total retransmitted packets in window
110. `syn_ack_ratio` — SYN to ACK ratio across window
