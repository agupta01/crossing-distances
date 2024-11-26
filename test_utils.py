from utils import decode_crosswalk_id, get_crosswalk_id


def test_encode_decode_crosswalk_id():
    lat = 37.724522
    long = -122.503095

    # encode
    crosswalk_id = get_crosswalk_id(lat, long)
    assert crosswalk_id == "37724522N_122503095W"

    # decode
    decoded_coord = decode_crosswalk_id(crosswalk_id)
    assert decoded_coord == (37.724522, -122.503095)
