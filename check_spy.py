from ib_insync import *

ib = IB()
try:
    print("--> Connecting to TWS...")
    ib.connect('127.0.0.1', 7497, clientId=999) # Using ID 999 to avoid conflict
    print("[SUCCESS] Connected.")

    print("--> Requesting SPY Data...")
    spy = Stock('SPY', 'SMART', 'USD')
    
    # 1. Check if contract resolves
    details = ib.reqContractDetails(spy)
    if not details:
        print("[FAIL] Could not find SPY contract. TWS settings issue?")
    else:
        print(f"[OK] Contract Found: {details[0].contract.localSymbol}")

    # 2. Check Historical Data (What the Guard uses)
    bars = ib.reqHistoricalData(
        spy, endDateTime='', durationStr='1800 S',
        barSizeSetting='1 min', whatToShow='TRADES', useRTH=True
    )

    if not bars:
        print("\n[CRITICAL FAILURE] SPY Data is EMPTY.")
        print("Reason: You likely do not have a real-time data subscription for SPY in Paper Trading.")
        print("Fix: You must subscribe to 'US Equity and Options Add-On Streaming Bundle' in Account Management.")
    else:
        print(f"\n[SUCCESS] Received {len(bars)} bars for SPY.")
        print(f"Latest Close: ${bars[-1].close}")

except Exception as e:
    print(f"\n[ERROR] {e}")
finally:
    ib.disconnect()