# SCP Scope Direct firmware

Universal direct-stream firmware for STM32F103 / STM32duino and Arduino Leonardo / ATmega32U4.

The MCU continuously samples ADC channels and sends chronological SCP1 packets in small transport blocks. Trigger, history, pre-trigger, Auto/Normal behavior and display timebase are owned by the Windows client.

The standard-rate macro is `SCOPE_STANDARD_RATE_HZ`. The command `PROFILE NORMAL` selects that rate; legacy `PROFILE ENGINE` is still accepted as an alias for compatibility.


## Channel support

- STM32F103: 2 or 3 channels by default, with optional **4-channel** acquisition by setting `SCOPE_CHANNEL_COUNT 4`. The fourth default input is `PA3`.
- Arduino Leonardo / ATmega32U4: remains at 2 channels.

The 4-channel STM32 profile intentionally uses more conservative sample-rate limits so its transport bandwidth stays close to the established 3-channel mode. Oversized serial command lines are now discarded atomically, and repeating the same rate/profile no longer restarts acquisition unnecessarily.
