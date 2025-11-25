import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from retrieval.flexible_retriever import FlexibleRetriever

class TestMultiImageSupport(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_images_client = MagicMock()
        self.retriever = FlexibleRetriever(
            client=self.mock_client,
            collection_name="test_code",
            swe_images_collection="swe_images",
            images_client=self.mock_images_client
        )
        
        # Mock embedding generator to avoid loading real models
        self.retriever.models['jina'] = MagicMock()
        self.retriever.models['jina'].embed_query.return_value = MagicMock(tolist=lambda: [0.1]*768)

    def test_fetch_visual_descriptions(self):
        # Mock scroll response with 2 images
        mock_point1 = MagicMock()
        mock_point1.payload = {"vlm_description": "desc1"}
        mock_point2 = MagicMock()
        mock_point2.payload = {"vlm_description": "desc2"}
        
        self.mock_images_client.scroll.return_value = ([mock_point1, mock_point2], None)
        
        descs = self.retriever._fetch_visual_descriptions("instance_1")
        self.assertEqual(len(descs), 2)
        self.assertEqual(descs, ["desc1", "desc2"])

    def test_retrieve_augment_mode(self):
        # Mock descriptions
        self.retriever._fetch_visual_descriptions = MagicMock(return_value=["desc1", "desc2"])
        
        # Mock query_points to return something
        self.mock_client.query_points.return_value.points = []
        
        self.retriever.retrieve(
            query="fix bug",
            instance_id="instance_1",
            strategy=["jina"],
            visual_mode="augment"
        )
        
        # Check if embed_query was called with augmented text
        # The query should be "fix bug desc1 desc2"
        self.retriever.models['jina'].embed_query.assert_called_with("fix bug desc1 desc2")

    def test_retrieve_fusion_mode(self):
        # Mock descriptions
        self.retriever._fetch_visual_descriptions = MagicMock(return_value=["desc1", "desc2"])
        
        # Mock query_points
        self.mock_client.query_points.return_value.points = []
        
        self.retriever.retrieve(
            query="fix bug",
            instance_id="instance_1",
            strategy=["jina"],
            visual_mode="fusion"
        )
        
        # Check if embed_query was called 3 times: 
        # 1. "fix bug"
        # 2. "desc1"
        # 3. "desc2"
        calls = self.retriever.models['jina'].embed_query.call_args_list
        # Note: order might vary but we expect these calls
        args = [c[0][0] for c in calls]
        self.assertIn("fix bug", args)
        self.assertIn("desc1", args)
        self.assertIn("desc2", args)
        self.assertEqual(len(calls), 3)

if __name__ == '__main__':
    unittest.main()
