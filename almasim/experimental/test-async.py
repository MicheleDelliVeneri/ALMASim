import asyncio
import asyncssh
import socket

async def test_ssh_connection():
    try:
        async with asyncssh.connect(
            '172.16.14.132',
            username='astro',
            client_keys=['/Users/michele/.ssh/astro_workstation_rsa'],
            known_hosts=None,
            family=socket.AF_INET,  # Force IPv4
        ) as conn:
            result = await conn.run('echo "SSH connection successful"', check=True)
            print(result.stdout)
    except Exception as e:
        print(f"SSH connection failed: {e}")

# Use asyncio.run() instead of get_event_loop().run_until_complete()
asyncio.run(test_ssh_connection())