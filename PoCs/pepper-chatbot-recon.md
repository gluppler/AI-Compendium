# Pepper Chatbot Reconnaissance — Chipotle

## Overview

Pepper is Chipotle's customer-facing virtual assistant, built on **Amelia** (IPsoft/Amelia AI platform). It lives on the contact-us page and handles customer inquiries.

## Surface

### Entry Points

| URL | Purpose |
|-----|---------|
| `https://www.chipotle.com/contact-us` | Landing page with Pepper embed |
| `https://amelia.chipotle.com/Amelia/ui/chipotle/chat?embed=iframe` | Amelia widget iframe source |
| `https://amelia.chipotle.com/Amelia/ui/chipotle/config.json` | Client config (408 KB) |
| `https://amelia.chipotle.com/Amelia/ui/chipotle/assets/amelia.js` | Amelia JS client bundle |

### Backend

| Endpoint | Description |
|----------|-------------|
| `https://services.chipotle.com` | API gateway (proxies to Amelia backend) |
| `https://services.chipotle.com/149e9513-01fa-4fb0-aad4-566afd725d1b/2d206a39-8ed7-437e-a3be-862e0f06eea3/p.js` | DataDome KPSDK bot protection script |

### Config Key Findings (`config.json`)

```
client.domainCode = "chipotle"
client.canJoinDefaultDomain = true
ui.parentFrameUrl = "https://www-dev.chipotle.com|https://www-qa.chipotle.com|..."
ui.hideLogin = true
ui.hideLogout = true
ui.browserTabTitle = "Pepper Chat"
```

The config contains **no direct API endpoint URLs**. It is purely client-side UI settings, SVG icons (base64), CSS overrides, and localization strings. Actual API endpoints are resolved at runtime by the Amelia JS client.

## Architecture

```
User Browser
  │
  ├── https://www.chipotle.com/contact-us
  │     └── <iframe> → amelia.chipotle.com/Amelia/ui/chipotle/chat?embed=iframe
  │           └── Amelia SPA
  │                 ├── config.json → UI settings, domainCode
  │                 ├── amelia.js   → client logic, API bindings
  │                 └── postMessage → parent window (origin validated against parentFrameUrl)
  │
  ├── WebSocket → wss://amelia.chipotle.com/Amelia/api/sock/{serverId}/{sessionId}/websocket
  │     └── STOMP protocol (all chat messages)
  │
  └── XHR to services.chipotle.com (behind DataDome KPSDK)
        └── Proxied to Amelia backend (IPsoft cloud)
```

## Real-Time Protocol: STOMP over WebSocket

**Pepper does NOT use HTTP POST for messages.** The Amelia client establishes a persistent WebSocket connection and communicates via the STOMP (Simple Text Oriented Message Protocol) protocol.

### Connection Establishment

1. Amelia JS loads config, creates a session
2. JS opens a WebSocket to:
   `wss://amelia.chipotle.com/Amelia/api/sock/{serverId}/{sessionId}/websocket`
3. STOMP frames are sent as WebSocket **binary** messages

### STOMP Frame Format

Each frame follows the STOMP spec:
```
COMMAND\nheader1:value1\nheader2:value2\n\nbody\x00
```

### Session Flow

#### 1. CONNECT
```
CONNECT
accept-version:1.1,1.0
host:amelia.chipotle.com
X-CSRF-TOKEN:{token}
X-Amelia-Session-Auth:{auth_token}

\x00
```

#### 2. CONNECTED (server response)
```
CONNECTED
version:1.1
session:{sessionId}
server:Amelia/{version}

\x00
```

#### 3. SUBSCRIBE (to reply queue)
```
SUBSCRIBE
destination:/queue/session.{sessionId}
id:sub-0

\x00
```

#### 4. SEND (user message)
```
SEND
destination:/amelia/session.in
content-type:application/json;charset=UTF-8
X-Amelia-Session-Id:{sessionId}
X-Amelia-Conversation-Id:{conversationId}
X-Amelia-Message-Type:InboundUserUtteranceMessage
X-Amelia-Timestamp:{unix_ms_timestamp}

{"messageText":"your text","attributes":{"formInputAttributes":{...}},"secure":false,"offTheRecord":false}\x00
```

#### 5. MESSAGE (server reply)
```
MESSAGE
destination:/queue/session.{sessionId}
message-id:{messageId}
content-type:application/json;charset=UTF-8
X-Amelia-Session-Id:{sessionId}
X-Amelia-Conversation-Id:{conversationId}
X-Amelia-Message-Type:OutboundTextMessage
X-Amelia-Timestamp:{timestamp}

{"messageText":"Pepper's reply","intentName":"...","intentConfidence":1.0,"contextVariables":{"name":"...",...}}\x00
```

### Message Body JSON Structure

**User message:**
```json
{
  "messageText": "your message here",
  "attributes": {
    "formInputAttributes": {},
    "allowIntentDetectionInFormInput": true
  },
  "secure": false,
  "offTheRecord": false
}
```

**When consent is accepted ("I Agree, Start"):**
```json
{
  "messageText": "I Agree, Start",
  "attributes": {
    "formInputAttributes": {
      "SelectionField": [
        {
          "name": "Pepper_ToS",
          "value": "Yes"
        }
      ]
    },
    "allowIntentDetectionInFormInput": true
  },
  "secure": false,
  "offTheRecord": false
}
```

**Pepper response (key fields):**
```json
{
  "messageText": "...",
  "intentName": "Chipotle_Generic_Greeting",
  "intentConfidence": 1.0,
  "contextVariables": { ... }
}
```

### Key Observations

- Consent is **not** a separate endpoint — "I Agree" is sent as a normal `InboundUserUtteranceMessage` with a `SelectionField` attribute
- The WebSocket connection is reused for multiple messages (session persists)
- `Conversation-Id` and `Session-Id` remain constant across messages in the same session
- Messages are STOMP frames, **not** raw WebSocket text — parsing the STOMP envelope is required
- `X-Amelia-Message-Type` distinguishes inbound (`InboundUserUtteranceMessage`) from outbound (`OutboundTextMessage`) messages
- Replies arrive on the subscribed queue destination `/queue/session.{sessionId}`

## Consent / ToC Gate

Pepper requires users to accept a terms-of-consent dialog before any conversation can begin:

> "Hey! I'm Pepper, your automated assistant chatbot. Before we begin, please know that chats may be monitored, recorded, used, and disclosed by Chipotle, the vendor powering this chatbot, and other third parties as described in our Privacy Policy"

Buttons: **I Agree** | **I Disagree**

Consent is sent as a regular STOMP `SEND` with `messageText: "I Agree, Start"` and a `SelectionField` attribute with `value: "Yes"`. The session is not authorized for real queries until this consent message is sent.

## Authentication

- No visible login flow (`hideLogin: true`)
- Session tokens are obtained by the Amelia JS client during initialization
- The STOMP CONNECT frame includes `X-CSRF-TOKEN` and `X-Amelia-Session-Auth` headers
- Additional session/authentication cookies may be required from `services.chipotle.com`

## Notes for Attack Tooling

1. **Protocol**: Pepper uses STOMP-over-WebSocket, not HTTP. The PoC at `PoCs/rust-jailbreak/` now includes a `stomp.rs` STOMP client module. Use `STOMP_URL`, `STOMP_CONNECT_HEADERS`, `STOMP_MESSAGE_HEADERS` env vars (see `PoCs/README.md`).
2. **DataDome**: The KPSDK script at `p.js` performs client-side fingerprinting. Automated requests without valid DataDome tokens will be blocked. Solution: extract cookies + DataDome token from a real browser session.
3. **CORS**: `destinationFrameUrl: *` allows any origin to receive postMessages from the iframe — potentially exploitable for XSS-like interception.
4. **Session reuse**: WebSocket connections may be short-lived. Set `STOMP_REUSE=1` to keep the connection open across attack iterations.
5. **Consent**: Must send "I Agree, Start" with consent attribute before any real message. The STOMP module sends the raw JSON body; configure the consent message via the prompt parameter.

## Methodology

This data was gathered by:
1. Loading `https://www.chipotle.com/contact-us` in browser
2. Opening DevTools → Network tab → WS filter
3. Sending messages to Pepper
4. Exporting HAR from the iframe context (contains SockJS/WebSocket traffic)
5. Parsing WebSocket frames to extract STOMP protocol details
