# SCP Scope Direct firmware

Universal direct-stream firmware for STM32F103 / STM32duino and Arduino Leonardo / ATmega32U4.

The MCU continuously samples ADC channels and sends chronological SCP1 packets in small transport blocks. Trigger, history, pre-trigger, Auto/Normal behavior and display timebase are owned by the Windows client.

The standard-rate macro is `SCOPE_STANDARD_RATE_HZ`. The command `PROFILE NORMAL` selects that rate; legacy `PROFILE ENGINE` is still accepted as an alias for compatibility.
