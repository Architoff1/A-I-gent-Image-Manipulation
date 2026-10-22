from PIL import Image

def test_mask_convention_and_composite():
    original = Image.new("RGB", (4, 4), (10, 20, 30))
    generated = Image.new("RGB", (4, 4), (200, 210, 220))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    final = Image.composite(generated, original, mask)
    assert final.getpixel((0, 0)) == original.getpixel((0, 0))
    assert final.getpixel((1, 1)) == generated.getpixel((1, 1))
